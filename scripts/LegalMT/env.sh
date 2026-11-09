#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    _LEGALMT_ENV_EXECUTED=1
else
    _LEGALMT_ENV_EXECUTED=0
fi

LEGALMT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LEGALMT_REPO_ROOT

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$LEGALMT_REPO_ROOT/.venv-gpu}"

export LEGALMT_ARTIFACTS_ROOT="${LEGALMT_ARTIFACTS_ROOT:-$LEGALMT_REPO_ROOT/../legalmt-artifacts}"

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
export TMPDIR="${TMPDIR:-/tmp}"

export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    echo "[env.sh] note: no env at $UV_PROJECT_ENVIRONMENT yet; build it with scripts/LegalMT/setup_gpu_env.sh" >&2
fi

if [[ "$_LEGALMT_ENV_EXECUTED" == 1 ]]; then
    echo "env.sh was executed, not sourced: nothing was changed in your shell."
    echo "Use: source ${BASH_SOURCE[0]}"
    echo "It would set:"
    for _v in LEGALMT_REPO_ROOT UV_PROJECT_ENVIRONMENT LEGALMT_ARTIFACTS_ROOT HF_HOME UV_CACHE_DIR TMPDIR \
        VLLM_NO_USAGE_STATS VLLM_USE_FLASHINFER_SAMPLER HF_HUB_DISABLE_TELEMETRY; do
        echo "  $_v=${!_v}"
    done
fi
unset _LEGALMT_ENV_EXECUTED _v
