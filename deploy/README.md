# The running deployment

Copies of what is actually installed on the orchestrator box, so that a change to
it is reviewable and a rebuild is not archaeology. They are *copies*: editing a
file here changes nothing until it is put back (`sudo install -m 644 …`,
`systemctl daemon-reload`).

    orchestrator.service   /etc/systemd/system/kunlun.service
    h3-tunnel@.service     /etc/systemd/system/h3-tunnel@.service   (disabled)
    h3-tunnel.sh           /home/ubuntu/kunlun/h3-tunnel.sh         (disabled)
    game_tunnel.sh         /home/ubuntu/kunlun/game_tunnel.sh       (the old laptop-era script)

`local_proxy.sh` is the exception: it runs **on the laptop**, not on the box.

## Three hosts, and which direction each door opens

    laptop ──ssh -L 8100───► orchestrator 172.31.42.3 ──http──► P5-1 172.31.45.68:30010
                               127.0.0.1:8100  control API              P5-2 172.31.33.181:30010
                               0.0.0.0:8101    assets  ◄──http── (the GPU boxes fetch frames)

Two sockets on the orchestrator, and the split is the point. The control API
creates sessions and spends GPU slots and has no auth of its own, so it binds
loopback and the only way in is the tunnel from the laptop. The asset server is a
separate app whose whole routing table is one read-only `StaticFiles` mount, and
that is the only thing bound where the other instances can see it -- because
`H3_TRANSPORT=http` has the GPU boxes fetch the conditioning frame themselves
rather than having it pushed to them over ssh.

Verified from P5-1: `GET /<session>/kf-….png` → 200, `POST` → 405,
`GET :8100/healthz` → no route.

## The one tunnel that is on: laptop → orchestrator

    bash deploy/local_proxy.sh up | status | log | down

A launchd agent (`~/Library/LaunchAgents/com.h3game.proxy.plist`) holding
`-L 8100:127.0.0.1:8100`. It is supervised because an unsupervised one demonstrably
does not survive this laptop: `pmset -g log` showed four `Maintenance Sleep`
cycles inside ten minutes on battery, and the default route is a VPN (`utun4`)
whose reconnects drop every TCP connection under it. ssh notices and exits within
~45s (`ServerAliveInterval=15` × 3); `KeepAlive` is the half that was missing.
Measured recovery from a `kill -9`: **9s**, with no one watching.

Two details are macOS-specific and both come from the same fact -- a
launchd-spawned process has no TCC grant, so it cannot read `~/Documents` at all:

* the plist execs `/usr/bin/ssh` directly, with the flags inline, rather than
  running a wrapper script out of the repo;
* the key is copied once to `~/.ssh/henanwan-us-east-2.pem` (mode 600) because the
  original lives under `~/Documents/account/…`. Pointing the agent at the original
  fails with `Operation not permitted`, twice, and then launchd throttles it.

`status` prints launchd's `runs` counter: a number that climbs on its own is a
flapping network, not a broken config.

## The tunnels on the box are off

`h3-tunnel@30010` / `@30011` forwarded a local port to each GPU box's loopback,
back when SGLang bound `127.0.0.1`. Both now bind `0.0.0.0`, `H3_REPLICAS` names
their private addresses, and `kunlun.service` no longer waits on either unit.

The security group was never the obstacle. A direct connection to a GPU port was
*refused* -- a kernel RST -- and a security group that denies traffic drops it
silently, so the refusal proved the path was already allowed.

## Environment

`/home/ubuntu/kunlun/.env`, read by `EnvironmentFile=`; not in git, it holds the
Bedrock token. The keys that decide the topology above:

    H3_TRANSPORT=http
    H3_REPLICAS=P5-1=172.31.45.68:30010,P5-2=172.31.33.181:30010
    ASSETS_PORT=8101
    ASSETS_DIR=/home/ubuntu/kunlun-data/assets
    # INTERNAL_BASE_URL=  -- unset: worked out from the routing table at startup

`INTERNAL_BASE_URL` is what the GPU boxes are told to fetch from. Left unset it is
`http://<the address on the interface that routes outward>:ASSETS_PORT`, which on
this box is `172.31.42.3` -- asked of the routing table rather than of `ip addr`,
because the docker bridges at 172.17.0.1 and 172.18.0.1 are also addresses of this
host and neither is reachable from anywhere else. `/healthz` reports the answer as
`assets_for_gpu`.
