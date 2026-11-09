#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$SCRIPT_DIR/env.sh"

echo "[setup] repo:  $REPO_ROOT"
echo "[setup] env:   $UV_PROJECT_ENVIRONMENT"
echo "[setup] cache: $UV_CACHE_DIR"
df -h "$REPO_ROOT"

uv sync --project "$REPO_ROOT" --group gpu --locked "$@"

echo "[setup] installed versions:"
uv run --project "$REPO_ROOT" --group gpu --locked python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

names = ["torch", "torchvision", "vllm", "transformers", "tokenizers", "peft", "trl",
         "accelerate", "bitsandbytes", "datasets", "huggingface_hub", "hf_xet",
         "sentence-transformers", "sacrebleu", "pandas", "pytest"]
for name in names:
    try:
        print(f"  {name:22s} {version(name)}")
    except PackageNotFoundError:
        print(f"  {name:22s} NOT INSTALLED")

import torch  # importing torch does not touch the GPU

print(f"  torch.version.cuda     {torch.version.cuda}")
# is_available() only asks the driver how many devices exist; it allocates no GPU memory.
print(f"  cuda available         {torch.cuda.is_available()}")
PY

du -sh "$UV_PROJECT_ENVIRONMENT"
df -h "$REPO_ROOT"
