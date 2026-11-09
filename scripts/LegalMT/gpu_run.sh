#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ $# -lt 1 ]]; then
    echo "usage: $0 <python-script> [args...]" >&2
    exit 2
fi
source "$SCRIPT_DIR/env.sh"
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "[gpu_run] $(date '+%F %T') GPU state:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | sed 's/^/  gpu: /'
    apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
    if [[ -z "$apps" ]]; then
        echo "  no compute processes on the GPU"
    else
        while IFS=, read -r pid pname mem; do
            owner="$(ps -o user= -p "${pid// /}" 2>/dev/null || echo '?')"
            echo "  process: pid=${pid// /} user=${owner:-?} mem=${mem# } name=${pname# }"
        done <<< "$apps"
    fi
else
    echo "[gpu_run] nvidia-smi not found: no GPU state to print."
fi
rc=0
uv run --project "$REPO_ROOT" --group gpu python "$@" || rc=$?
echo "[gpu_run] $(date '+%F %T') exit code $rc"
exit "$rc"
