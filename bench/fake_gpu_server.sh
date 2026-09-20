#!/usr/bin/env bash
# A throwaway orchestrator for testing story-engine logic without a GPU.
#
# Same code and the same Bedrock credentials as the live service, but the fake
# video backend and its own DATA_DIR, so a test run neither needs SGLang to be up
# nor leaves sessions in the player's history. Port 8109, so it can run alongside
# the real thing on 8100.
set -euo pipefail

set -a
. /home/ubuntu/kunlun/.env
set +a

export GPU_BACKEND=fake FAKE_GPU=1 FAKE_GPU_LATENCY_S=1
export DATA_DIR=/tmp/pa ASSETS_DIR=/tmp/pa/assets
export PUBLIC_BASE_URL=http://127.0.0.1:8109/assets

cd /home/ubuntu/kunlun/services/orchestrator
# A pidfile rather than `pkill -f`, because every pattern specific enough to find
# this server also appears in the ssh command line doing the killing -- so pkill
# kills its own shell and the restart reports 255 with nothing restarted. `exec`
# keeps this pid, so the file stays correct.
echo $$ > /tmp/pa.pid
exec /home/ubuntu/kunlun/.venv/bin/python -m uvicorn app.main:app \
  --host 127.0.0.1 --port 8109
