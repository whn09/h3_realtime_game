#!/usr/bin/env bash
# Hold one forward open: local $1 -> that replica's loopback 30010.
#
# A script on disk rather than an inline `bash -c` in the unit, because systemd's
# ExecStart parser eats the `;;` of a case statement -- it splits on `;` before the
# shell ever sees the line, which fails as "unexpected EOF" and reads like a shell
# bug rather than a unit-file bug.
#
# DISABLED, and kept only as the way back. Both GPU boxes now bind 0.0.0.0, so
# `H3_REPLICAS` names their private addresses and there is nothing to forward:
#
#   systemctl disable --now h3-tunnel@30010 h3-tunnel@30011
#
# To bring it back -- a box rebound to loopback, a VPC that stops routing -- start
# those two units and set H3_REPLICAS back to `P5-1=127.0.0.1:30010,...`.
#
# The security group was never what made this necessary: a direct connection to a
# GPU port was refused (a kernel RST) rather than dropped, and an SG that denies
# traffic drops it silently. The bind address was the whole of it.
set -u

case "$1" in
  30010) ALIAS=P5-1 ;;
  30011) ALIAS=P5-2 ;;
  *) echo "no replica mapped to local port $1" >&2; exit 64 ;;
esac

# -N: forward and nothing else. A tunnel that also allocates a pty dies
# differently -- the remote shell can exit and take the forward with it, and the
# failure then surfaces as an HTTP timeout three layers away.
# ExitOnForwardFailure: without it ssh connects, fails to bind the local port
# because a stale tunnel still holds it, and then sits there looking healthy while
# every request goes nowhere.
exec /usr/bin/ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "$1:127.0.0.1:30010" "$ALIAS"
