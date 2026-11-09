#!/usr/bin/env bash
# Build the COMET scoring env from tools/comet/uv.lock and fetch the pinned COMET checkpoint.
#
#     scripts/LegalMT/setup_comet_env.sh [uv sync args, e.g. --reinstall]
#
# The env goes to env_dir in tools/comet/pyproject.toml (default .venv-comet) or $LEGALMT_COMET_ENV.
# MIN_FREE_GB (default 10) is the free space that must remain after the build. -h / --help prints this text.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMET_PROJECT="$REPO_ROOT/tools/comet"

source "$SCRIPT_DIR/env.sh"

gpu_env="$(env -u UV_PROJECT_ENVIRONMENT bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${UV_PROJECT_ENVIRONMENT:-}"' \
    _ "$SCRIPT_DIR/env.sh")"
default_env="$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["tool"]["legalmt-comet"]["env_dir"])' \
    "$COMET_PROJECT/pyproject.toml")"
export UV_PROJECT_ENVIRONMENT="${LEGALMT_COMET_ENV:-$default_env}"
case "$UV_PROJECT_ENVIRONMENT" in /*) ;; *) UV_PROJECT_ENVIRONMENT="$REPO_ROOT/$UV_PROJECT_ENVIRONMENT"; export UV_PROJECT_ENVIRONMENT ;; esac
if [[ -z "${UV_PROJECT_ENVIRONMENT%/}" || -z "$gpu_env" \
      || "$(realpath -m "$UV_PROJECT_ENVIRONMENT")" == "$(realpath -m "$gpu_env")" ]]; then
    echo "[setup-comet] refusing env '$UV_PROJECT_ENVIRONMENT': empty or the LegalMT GPU env ('$gpu_env')" >&2
    exit 2
fi

echo "[setup-comet] project: $COMET_PROJECT"
echo "[setup-comet] env:     $UV_PROJECT_ENVIRONMENT"
echo "[setup-comet] cache:   $UV_CACHE_DIR"
echo "[setup-comet] HF_HOME: $HF_HOME"

MIN_FREE_GB="${MIN_FREE_GB:-10}"
NEED_GB="${NEED_GB:-4}"
existing_ancestor() {
    local p="$1"
    while [[ ! -e "$p" && "$p" != "/" ]]; do p="$(dirname "$p")"; done
    echo "$p"
}
for target in "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR" "$HF_HOME"; do
    where="$(existing_ancestor "$target")"
    avail_gb="$(df --output=avail -BG "$where" | tail -n 1 | tr -dc '0-9')"
    mount="$(df --output=target "$where" | tail -n 1)"
    echo "[setup-comet] $target: ${avail_gb} GB free on $mount"
    if (( avail_gb - NEED_GB < MIN_FREE_GB )); then
        echo "[setup-comet] $mount has ${avail_gb} GB free; building needs about ${NEED_GB} GB and must leave ${MIN_FREE_GB} GB free. Aborting." >&2
        exit 1
    fi
done

uv sync --project "$COMET_PROJECT" --locked "$@"

echo "[setup-comet] installed versions:"
uv run --project "$COMET_PROJECT" --locked python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for name in ["unbabel-comet", "torch", "transformers", "tokenizers", "numpy", "pytorch-lightning",
             "torchmetrics", "huggingface-hub", "sentencepiece", "pandas", "scipy"]:
    try:
        print(f"  {name:20s} {version(name)}")
    except PackageNotFoundError:
        print(f"  {name:20s} NOT INSTALLED")

import torch  # importing torch does not touch the GPU

print(f"  torch.version.cuda   {torch.version.cuda}")
# is_available() only asks the driver how many devices exist; it allocates no GPU memory.
print(f"  cuda available       {torch.cuda.is_available()}")
PY

uv run --project "$COMET_PROJECT" --locked python "$SCRIPT_DIR/comet_score.py" --download

du -sh "$UV_PROJECT_ENVIRONMENT"
df -h "$(existing_ancestor "$UV_PROJECT_ENVIRONMENT")" "$(existing_ancestor "$HF_HOME")"
