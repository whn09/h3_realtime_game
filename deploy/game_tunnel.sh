#!/usr/bin/env bash
# Bring each replica's loopback HTTP port to a distinct port on THIS laptop, and keep it there.
#
#   bash game_tunnel.sh H100-A H100-B     # -> 127.0.0.1:30010 and 127.0.0.1:30011
#   bash game_tunnel.sh down
#   bash game_tunnel.sh status
#
# The arguments are ssh host aliases (~/.ssh/config), not hostnames, for the same reason the rest of
# this repo takes them that way: the key path, the user and the long ec2 DNS name belong in one place.
#
# WHY A FILE OF REPLICAS. game_client.py needs two things per replica that a port number alone does
# not carry: where to POST, and how to copy the finished mp4 back. The server writes the clip to its
# OWN filesystem and the response carries `file_path` -- there is no download endpoint on /v1/videos
# (nothing in this repo uses one, and the API is submit + poll). So the client scp's it, and it needs
# the ssh alias to do that. The mapping alias -> local port is written here and read there.
#
# WHY -N AND NOT A LOGIN SHELL. A tunnel that also allocates a pty dies differently: the remote shell
# can exit, or a `logout` can take the forward with it, and the failure looks like an HTTP timeout
# three layers away. -N forwards and nothing else.
set -uo pipefail

BASE_PORT=${BASE_PORT:-30010}
REMOTE_PORT=${REMOTE_PORT:-30010}
STATE=${STATE:-${TMPDIR:-/tmp}/h3game}
REPLICAS=$STATE/replicas

mkdir -p "$STATE"

case "${1:-}" in
  down)
    if [ -d "$STATE" ]; then
      for p in "$STATE"/*.pid; do
        [ -e "$p" ] || continue
        pid=$(cat "$p")
        kill "$pid" 2>/dev/null && echo "closed tunnel pid $pid ($(basename "$p" .pid))"
        rm -f "$p"
      done
      rm -f "$STATE"/cm-*     # the control sockets die with their master; don't leave stale paths
    fi
    rm -f "$REPLICAS"
    exit 0
    ;;
  status)
    [ -f "$REPLICAS" ] || { echo "no tunnels (no $REPLICAS)"; exit 0; }
    while read -r alias endpoint; do
      # /health is not assumed: the check is that something answers the port at all. A 404 from a
      # live server is a pass here -- this is liveness of the FORWARD, not of the model.
      if curl -s -o /dev/null -m 3 "http://$endpoint/v1/videos" -X POST -d '{}' \
           -H 'Content-Type: application/json'; then
        echo "up    $alias  ->  $endpoint"
      else
        echo "DOWN  $alias  ->  $endpoint"
      fi
    done < "$REPLICAS"
    exit 0
    ;;
esac

[ $# -ge 1 ] || { echo "usage: game_tunnel.sh <ssh-alias> [<ssh-alias> ...] | down | status"; exit 2; }

: > "$REPLICAS"
port=$BASE_PORT
for alias in "$@"; do
  # -o ExitOnForwardFailure=yes: without it ssh connects, fails to bind the local port (because a
  # previous tunnel still holds it), and sits there looking healthy while every request goes to the
  # OLD replica. That is the one failure mode of this script worth spending a flag on.
  # ServerAlive*: an idle NAT or an ELB will drop a forward that carries no traffic between clips,
  # and a game that renders one beat a minute is exactly that shape.
  # ControlMaster, because the finished mp4 comes back over scp and a COLD scp is expensive:
  # measured us-east-2 -> this Mac, 946 KB, 5.4-6.1 s cold vs 2.65-3.4 s through this socket. The
  # handshake is half the cost of fetching a clip, and the tunnel is already holding a connection to
  # the same box, so the fix is to let game_client.py ride it instead of opening its own.
  ssh -N -f \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=yes -o "ControlPath=$STATE/cm-$alias" \
    -L "127.0.0.1:$port:127.0.0.1:$REMOTE_PORT" \
    "$alias" || { echo "FAILED to forward $alias -> 127.0.0.1:$port"; exit 1; }
  # -f backgrounds ssh itself, so $! is not its pid. Find it by the forward spec, which is unique.
  pid=$(pgrep -f "L 127.0.0.1:$port:127.0.0.1:$REMOTE_PORT" | head -1)
  echo "$pid" > "$STATE/$alias.pid"
  echo "$alias 127.0.0.1:$port" >> "$REPLICAS"
  echo "forwarded $alias  ->  127.0.0.1:$port  (pid $pid)"
  port=$((port + 1))
done

echo
echo "replicas written to $REPLICAS"
echo "next: python3 scripts/game_client.py --bench 5"
