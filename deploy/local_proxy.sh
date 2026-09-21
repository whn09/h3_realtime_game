#!/usr/bin/env bash
# The laptop -> orchestrator tunnel, supervised by launchd so it comes back by itself.
#
#   bash deploy/local_proxy.sh up       # install + start (survives sleep, VPN flaps, reboot)
#   bash deploy/local_proxy.sh status
#   bash deploy/local_proxy.sh log
#   bash deploy/local_proxy.sh down     # stop + uninstall, leaves nothing behind
#
# WHY THIS EXISTS. A bare `ssh -f -N -L 8100:...` does not survive this laptop, and the way it fails
# is silent: the browser stops loading and the orchestrator looks dead. Measured here, two things kill
# it on their own schedule and neither is a bug:
#
#   * `Maintenance Sleep` on battery -- `pmset -g log` showed four sleep/wake cycles inside ten
#     minutes, each long enough for the keepalive probes to trip;
#   * the default route is a VPN (`utun4`), and a VPN reconnect drops every TCP connection under it.
#
# ssh's own job is to notice and exit, and it does. Nothing was doing the other half -- starting it
# again. That is all launchd is here for: `KeepAlive` restarts it whenever it exits, for any reason,
# and `RunAtLoad` covers login and reboot. The forward is then down for the seconds between a failed
# probe and the restart, instead of until someone notices and asks.
#
# WHY THE AGENT RUNS ssh DIRECTLY, AND WHY THE KEY IS COPIED TO ~/.ssh. Both are the same macOS fact:
# a launchd-spawned process has no TCC grant, so it cannot read `~/Documents` at all -- the first
# version of this file pointed the agent at the repo copy of the script and at the pem under
# `~/Documents/account/...`, and launchd logged `Operation not permitted` twice before giving up. So:
# no wrapper script for launchd to read (the plist execs /usr/bin/ssh with the flags inline), and the
# key is copied once to `~/.ssh`, which is not TCC-protected. `up` does the copy with mode 600 and
# never overwrites an existing file.
#
# WHY NOT autossh. It would work, but it is another thing to install and it supervises by opening a
# second forwarded port pair to echo traffic through. launchd is already running, already handles
# wake-from-sleep, and already writes the exit reason to a log file.
#
# WHY THE TIMEOUTS ARE SHORT. 15s x 3: the forward is declared dead ~45s after the network goes away,
# where an unsupervised tunnel wanted to be patient (a long timeout is how you avoid a tunnel that
# cannot come back). With a supervisor, impatience is free -- dying fast is how it recovers fast.
set -uo pipefail

LABEL=${LABEL:-com.h3game.proxy}
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG=${LOG:-$HOME/Library/Logs/$LABEL.log}

# Where the key lives for the *agent*. Not the same path as the one a human uses from a terminal,
# which is the whole point of the TCC note above.
KEY=${KEY:-$HOME/.ssh/henanwan-us-east-2.pem}
KEY_SRC=${KEY_SRC:-/Users/henanwan/Documents/account/579019700964/henanwan/henanwan-us-east-2.pem}

HOST=${HOST:-ubuntu@ec2-13-59-29-243.us-east-2.compute.amazonaws.com}
PORT=${PORT:-8100}
REMOTE_PORT=${REMOTE_PORT:-8100}

DOMAIN="gui/$(id -u)"
FORWARD="127.0.0.1:$PORT:127.0.0.1:$REMOTE_PORT"

# -N -T: forward only. A tunnel that also allocates a pty dies differently -- the remote shell can
#   exit and take the forward with it, and that failure looks like an HTTP timeout three layers away
#   rather than like a dead tunnel.
# ExitOnForwardFailure: without it, a restart that races the dying process (which still holds the
#   local port) connects, fails to bind, and then sits there looking healthy while nothing is
#   forwarded. That is the one failure mode worth spending a flag on.
SSH_ARGS=(
  -i "$KEY" -N -T
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3
  -o TCPKeepAlive=yes
  -o ConnectTimeout=15
  -o StrictHostKeyChecking=accept-new
  -L "$FORWARD"
  "$HOST"
)

plist_args() {
  printf '    <string>%s</string>\n' /usr/bin/ssh "${SSH_ARGS[@]}"
}

case "${1:-status}" in
  up)
    if [ ! -f "$KEY" ]; then
      [ -f "$KEY_SRC" ] || { echo "no key at $KEY_SRC"; exit 1; }
      install -m 600 "$KEY_SRC" "$KEY" || exit 1
      echo "copied key -> $KEY (mode 600)"
    fi
    mkdir -p "$HOME/Library/LaunchAgents" "$(dirname "$LOG")"
    # ThrottleInterval, not the default 10s: if the box is unreachable (laptop offline, instance
    # stopped) this would otherwise retry six times a minute forever. 20s is still faster than a
    # human noticing.
    {
      cat <<'HEAD_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
HEAD_EOF
      printf '  <key>Label</key><string>%s</string>\n' "$LABEL"
      printf '  <key>ProgramArguments</key>\n  <array>\n'
      plist_args
      printf '  </array>\n'
      printf '  <key>RunAtLoad</key><true/>\n'
      printf '  <key>KeepAlive</key><true/>\n'
      printf '  <key>ThrottleInterval</key><integer>20</integer>\n'
      printf '  <key>StandardOutPath</key><string>%s</string>\n' "$LOG"
      printf '  <key>StandardErrorPath</key><string>%s</string>\n' "$LOG"
      printf '</dict>\n</plist>\n'
    } > "$PLIST"

    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    # Any hand-started tunnel still holding the port would make the agent exit on forward failure.
    pkill -f "L $FORWARD" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$PLIST" || { echo "bootstrap failed"; exit 1; }
    echo "installed $PLIST"
    for i in $(seq 1 20); do
      code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:$PORT/healthz")
      [ "$code" = "200" ] && { echo "up after ${i}s: 127.0.0.1:$PORT -> $HOST:$REMOTE_PORT"; exit 0; }
      sleep 1
    done
    echo "started, but /healthz did not answer in 20s -- see: bash $0 log"
    exit 1
    ;;

  down)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    pkill -f "L $FORWARD" 2>/dev/null
    rm -f "$PLIST"
    echo "stopped and removed $PLIST (the key copy at $KEY is left alone)"
    ;;

  status)
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      # These are the two things worth knowing: whether a process is alive right now, and how many
      # times it has been restarted -- a climbing `runs` is the signature of a flapping network
      # rather than of a broken config.
      launchctl print "$DOMAIN/$LABEL" \
        | grep -E "^[[:space:]]+(state|pid|runs|last exit code) =" | sed 's/^[[:space:]]*/  /'
    else
      echo "  launchd agent not installed (run: bash $0 up)"
    fi
    pgrep -f "L $FORWARD" >/dev/null && echo "  ssh forward alive" || echo "  no ssh forward"
    curl -s -o /dev/null -m 5 -w "  healthz http=%{http_code}\n" "http://127.0.0.1:$PORT/healthz"
    ;;

  log)
    tail -n "${2:-30}" "$LOG" 2>/dev/null || echo "no log at $LOG yet"
    ;;

  run)
    # Foreground, for debugging by hand. launchd does not use this path -- see the TCC note above.
    exec ssh "${SSH_ARGS[@]}"
    ;;

  *)
    echo "usage: $(basename "$0") up | down | status | log [n] | run"; exit 2
    ;;
esac
