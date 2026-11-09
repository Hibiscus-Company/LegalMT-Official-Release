#!/usr/bin/env bash
# Build the LegalMT Docker image: stage the build context, then run docker build.
#
#     docker/build_image.sh [options]
#
# Options:
#   --serve-config PATH   serving config (default configs/LegalMT/serve.yaml)
#   --infer-config NAME   replace its infer_config
#   --set KEY=VALUE       inference override baked into the image config (repeatable)
#   --model-dir DIR       serve these weights (a merged model) instead of the base snapshot in HF_HOME
#   --adapter DIR         bake in an unmerged LoRA adapter
#   --no-adapter          serve without an adapter (required with --model-dir when the config names one)
#   --licence-file FILE   model licence text, when the model dir has none
#   --team NAME           provider name in the licence NOTICE (default "Hibiscus Company")
#   --image NAME:TAG      image reference (default legalmt:<config_name>-<date>)
#   --context DIR         build context (default $LEGALMT_ARTIFACTS_ROOT/docker_build/<image>)
#   --base-image REF      Python base image
#   --uv-image REF        uv image
#   --force               replace an existing context folder
#   --stage-only          stage the context and print the docker build command
#   --allow-root-disk     no effect, kept for compatibility
#
# Environment: HF_HOME, LEGALMT_ARTIFACTS_ROOT, LEGALMT_PYTHON (a python with PyYAML for the config helpers)
set -euo pipefail
shopt -s inherit_errexit

PROG="build_image.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

die() {
    echo "$PROG: error: $*" >&2
    exit 1
}
note() { echo "[$PROG] $*"; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export TMPDIR="${TMPDIR:-/tmp}"
export LEGALMT_ARTIFACTS_ROOT="${LEGALMT_ARTIFACTS_ROOT:-$REPO_ROOT/../legalmt-artifacts}"
export PYTHONDONTWRITEBYTECODE=1

SERVE_CONFIG="$REPO_ROOT/configs/LegalMT/serve.yaml"
INFER_ARGS=()
SET_ARGS=()
MODEL_DIR=""
ADAPTER=""
NO_ADAPTER=0
LICENCE_FILE=""
TEAM="Hibiscus Company"
IMAGE=""
CONTEXT=""
BASE_IMAGE=""
UV_IMAGE=""
FORCE=0
STAGE_ONLY=0
ALLOW_ROOT=1

need_arg() { [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --serve-config) need_arg "$@"; SERVE_CONFIG=$2; shift 2 ;;
        --infer-config) need_arg "$@"; INFER_ARGS=(--infer-config "$2"); shift 2 ;;
        --set) need_arg "$@"; SET_ARGS+=(--set "$2"); shift 2 ;;
        --model-dir) need_arg "$@"; MODEL_DIR=$2; shift 2 ;;
        --adapter) need_arg "$@"; ADAPTER=$2; shift 2 ;;
        --no-adapter) NO_ADAPTER=1; shift ;;
        --licence-file | --license-file) need_arg "$@"; LICENCE_FILE=$2; shift 2 ;;
        --team) need_arg "$@"; TEAM=$2; shift 2 ;;
        --image) need_arg "$@"; IMAGE=$2; shift 2 ;;
        --context) need_arg "$@"; CONTEXT=$2; shift 2 ;;
        --base-image) need_arg "$@"; BASE_IMAGE=$2; shift 2 ;;
        --uv-image) need_arg "$@"; UV_IMAGE=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        --stage-only) STAGE_ONLY=1; shift ;;
        --allow-root-disk) ALLOW_ROOT=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done
if [[ $NO_ADAPTER == 1 ]]; then
    [[ -z $ADAPTER ]] || die "--adapter and --no-adapter contradict each other"
    SET_ARGS+=(--set "lora_adapter=null")
fi

if [[ $STAGE_ONLY == 0 ]]; then
    if ! command -v docker > /dev/null 2>&1; then
        die "docker is not installed on this machine.
  Install Docker Engine and the NVIDIA Container Toolkit (the image is about 12 GB; the build needs about 30 GB free
  under Docker's data-root), or run with --stage-only to prepare the build context here and build it on a Docker host."
    fi
    docker info > /dev/null 2>&1 || die "docker is installed but its daemon is not reachable (running? is $(id -un) in the docker group?)"
    DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2> /dev/null || true)
    if [[ -n $DOCKER_ROOT && -d $DOCKER_ROOT ]]; then
        free_gb=$(df -P -BG "$DOCKER_ROOT" | awk 'NR == 2 { gsub("G", "", $4); print $4 }')
        [[ $free_gb -ge 30 ]] || die "Docker data-root $DOCKER_ROOT has ${free_gb} GB free; the build needs about 30 GB (image about 12 GB plus layers)"
    fi
fi

if [[ -n ${LEGALMT_PYTHON:-} ]]; then
    PY=("$LEGALMT_PYTHON")
else
    command -v uv > /dev/null 2>&1 || die "uv not found; install it or set LEGALMT_PYTHON to a python with PyYAML"
    PY=(uv run --project "$REPO_ROOT" --quiet python)
fi
[[ -f $SERVE_CONFIG ]] || die "serving config not found: $SERVE_CONFIG"
SERVE_CONFIG=$(cd "$(dirname "$SERVE_CONFIG")" && pwd)/$(basename "$SERVE_CONFIG")
field() {
    "${PY[@]}" "$REPO_ROOT/docker/server.py" --config "$SERVE_CONFIG" "${INFER_ARGS[@]}" "${SET_ARGS[@]}" \
        --dump-config-field "$1"
}
CONFIG_NAME=$(field config_name) || die "could not resolve the serving/inference config (see the error above)"
BACKEND=$(field backend)
REPO_ID=$(field repo_id)
REVISION=$(field revision)
BASE_MODEL_DIR=$(field model_dir)
CFG_ADAPTER=$(field lora_adapter)
[[ $BACKEND != echo ]] || die "inference config $CONFIG_NAME uses the echo test backend; a submission needs a real model"

if [[ -n $MODEL_DIR ]]; then
    [[ -d $MODEL_DIR ]] || die "--model-dir $MODEL_DIR is not a directory"
    SRC_MODEL=$(cd "$MODEL_DIR" && pwd)
    MERGED=1
    MODEL_NAME=$(basename "$SRC_MODEL")
    SOURCE_DESC="model dir $SRC_MODEL (fine-tuned from ${REPO_ID:-?}@${REVISION:-?})"
else
    [[ -n $BASE_MODEL_DIR ]] || die "inference config $CONFIG_NAME names no model (repo_id + revision or local_path)"
    [[ -d $BASE_MODEL_DIR ]] || die "model snapshot $BASE_MODEL_DIR not found: fetch it with: hf download <repo_id> --revision <revision> (HF_HOME=$HF_HOME; see README.md) or pass --model-dir"
    SRC_MODEL=$(cd "$BASE_MODEL_DIR" && pwd)
    MERGED=0
    MODEL_NAME=${REPO_ID##*/}
    [[ -n $MODEL_NAME ]] || MODEL_NAME=$(basename "$SRC_MODEL")
    SOURCE_DESC=${REPO_ID:+$REPO_ID@$REVISION}
    SOURCE_DESC=${SOURCE_DESC:-local dir $SRC_MODEL}
fi
MODEL_NAME=$(printf '%s' "$MODEL_NAME" | tr -c 'A-Za-z0-9._-' '_')
[[ -f $SRC_MODEL/config.json ]] || die "$SRC_MODEL has no config.json"
compgen -G "$SRC_MODEL/*.safetensors" > /dev/null || die "$SRC_MODEL has no *.safetensors weights"
[[ -f $SRC_MODEL/tokenizer_config.json || -f $SRC_MODEL/tokenizer.json ]] || die "$SRC_MODEL has no tokenizer files"
if [[ $MERGED == 1 && -z $ADAPTER && -n $CFG_ADAPTER ]]; then
    die "inference config $CONFIG_NAME names lora_adapter=$CFG_ADAPTER, and --model-dir serves merged weights.
  If $SRC_MODEL already has that adapter merged in, applying it again would silently change every
  translation: pass --no-adapter. To really stack an adapter on these weights, pass --adapter DIR."
fi
ADAPTER=${ADAPTER:-$CFG_ADAPTER}
if [[ -n $ADAPTER ]]; then
    [[ -f $ADAPTER/adapter_config.json ]] || die "LoRA adapter $ADAPTER has no adapter_config.json"
    ADAPTER=$(cd "$ADAPTER" && pwd)
fi

if [[ -z $IMAGE ]]; then
    IMAGE="legalmt:$(printf '%s' "$CONFIG_NAME" | tr -c 'A-Za-z0-9_.-' '_')-$(date +%Y%m%d-%H%M)"
fi
SAFE_IMAGE=$(printf '%s' "$IMAGE" | tr '/:@' '___')
CONTEXT=${CONTEXT:-$LEGALMT_ARTIFACTS_ROOT/docker_build/$SAFE_IMAGE}
mkdir -p "$(dirname "$CONTEXT")"
CONTEXT="$(cd "$(dirname "$CONTEXT")" && pwd)/$(basename "$CONTEXT")"
if [[ $ALLOW_ROOT == 0 ]]; then
    mount_point=$(df -P "$(dirname "$CONTEXT")" | awk 'NR == 2 { print $6 }')
    [[ $mount_point != / ]] || die "context $CONTEXT is on the root filesystem ; use --context on a larger filesystem"
fi
case "$CONTEXT/" in
    "$REPO_ROOT"/*) die "the context must not be inside the repo ($REPO_ROOT)" ;;
esac
if [[ -e $CONTEXT ]]; then
    if [[ $FORCE == 1 ]]; then
        rm -rf -- "$CONTEXT"
    elif [[ -n $(ls -A "$CONTEXT" 2> /dev/null) ]]; then
        die "context $CONTEXT already exists; pass --force to replace it"
    fi
fi

note "staging $IMAGE into $CONTEXT"
mkdir -p "$CONTEXT"/{src,configs/LegalMT/infer,docker,models,licenses}
cp "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/uv.lock" "$CONTEXT/"
cp -R "$REPO_ROOT/src/." "$CONTEXT/src/"
find "$CONTEXT/src" \( -name __pycache__ -o -name '*.pyc' \) -prune -exec rm -rf {} +
cp "$REPO_ROOT"/configs/LegalMT/{paths,models,serve}.yaml "$CONTEXT/configs/LegalMT/"
cp "$REPO_ROOT"/configs/LegalMT/infer/*.yaml "$CONTEXT/configs/LegalMT/infer/"
cp "$SCRIPT_DIR"/{Dockerfile,server.py,translate.sh,build_image.sh,package.sh} "$CONTEXT/docker/"
cp "$SCRIPT_DIR/.dockerignore" "$CONTEXT/.dockerignore"

stage_files() {
    local src=$1 dst=$2 f real name n=0
    mkdir -p "$dst"
    for f in "$src"/*; do
        real=$(readlink -f -- "$f")
        [[ -f $real ]] || continue
        name=$(basename -- "$f")
        ln -- "$real" "$dst/$name" 2> /dev/null || cp -- "$real" "$dst/$name"
        n=$((n + 1))
    done
    echo "$n"
}
n_model=$(stage_files "$SRC_MODEL" "$CONTEXT/models/$MODEL_NAME")
note "model: $SOURCE_DESC -> /models/$MODEL_NAME ($n_model files)"
if [[ -n $ADAPTER ]]; then
    n_adapter=$(stage_files "$ADAPTER" "$CONTEXT/models/adapter")
    note "LoRA adapter: $ADAPTER -> /models/adapter ($n_adapter files)"
fi

find_licence() {
    local f
    for f in "$1"/License.txt "$1"/LICENSE.txt "$1"/LICENSE "$1"/LICENSE.md "$1"/license.txt; do
        if [[ -f $f ]]; then
            echo "$f"
            return 0
        fi
    done
    return 1
}
if [[ -n $LICENCE_FILE ]]; then
    [[ -f $LICENCE_FILE ]] || die "--licence-file $LICENCE_FILE not found"
    LIC_SRC=$LICENCE_FILE
elif LIC_SRC=$(find_licence "$SRC_MODEL"); then
    :
elif [[ -n $BASE_MODEL_DIR && -d $BASE_MODEL_DIR ]] && LIC_SRC=$(find_licence "$BASE_MODEL_DIR"); then
    :
else
    die "no licence file in $SRC_MODEL or in the base snapshot; pass --licence-file"
fi
if grep -qi "TENCENT HY COMMUNITY LICENSE" "$LIC_SRC"; then
    LICENCE="tencent-hy-community"
elif grep -q "Apache License" "$LIC_SRC"; then
    LICENCE="apache-2.0"
else
    LICENCE="other"
fi
cp -- "$LIC_SRC" "$CONTEXT/licenses/$(basename -- "$LIC_SRC")"
LIC_NAME=$(basename -- "$LIC_SRC")
{
    if [[ $LICENCE == tencent-hy-community ]]; then
        echo "Tencent HY is licensed under the Tencent HY Community License Agreement, Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent HY” are owned by Tencent or its affiliate."
        echo
        echo "This image contains the model $MODEL_NAME ($SOURCE_DESC), a Tencent HY Work distributed under the Tencent HY Community License Agreement; the full agreement is $LIC_NAME next to this file (/licenses/ in the image). Use of the Tencent HY Works is subject to the use restrictions of Sections 5(a) and 5(b) of that agreement, including its Acceptable Use Policy, and to its Territory clause."
        echo
        echo "This translation system is provided by $TEAM for the VLSP 2026 shared task. Tencent is not affiliated with, associated with, sponsoring, or endorsing it."
    elif [[ $LICENCE == apache-2.0 ]]; then
        echo "This image contains the model $MODEL_NAME ($SOURCE_DESC), licensed under the Apache License, Version 2.0; the licence text is $LIC_NAME next to this file (/licenses/ in the image)."
        echo
        echo "This translation system is provided by $TEAM for the VLSP 2026 shared task."
    else
        echo "This image contains the model $MODEL_NAME ($SOURCE_DESC); its licence is $LIC_NAME next to this file (/licenses/ in the image)."
    fi
    if [[ $MERGED == 1 ]]; then
        echo
        echo "Modified files: the model weights under /models/$MODEL_NAME were changed by $TEAM (fine-tuned from ${REPO_ID:-the base model}${REVISION:+@$REVISION})."
    fi
    if [[ -n $ADAPTER ]]; then
        echo
        echo "The LoRA adapter under /models/adapter was trained by $TEAM on top of ${REPO_ID:-the base model}; it is a modification of the base model's weights."
    fi
} > "$CONTEXT/licenses/NOTICE"
note "licence: $LICENCE ($LIC_NAME) + NOTICE"
[[ $LICENCE != other ]] || echo "$PROG: WARNING: unrecognised model licence in $LIC_SRC: check its redistribution terms before submitting" >&2

WIC=(--write-image-config "$CONTEXT/configs/LegalMT/serve.image.yaml" --model-local-path "/models/$MODEL_NAME"
    --meta "image=$IMAGE" --meta "built_at=$(date -Iseconds)" --meta "model_name=$MODEL_NAME"
    --meta "model_source=$SOURCE_DESC" --meta "licence=$LICENCE")
if [[ $MERGED == 1 ]]; then
    WIC+=(--merged-model --meta "base_model=${REPO_ID:-?}@${REVISION:-?}")
fi
if [[ -n $ADAPTER ]]; then
    WIC+=(--adapter-path /models/adapter --meta "adapter_source=$ADAPTER")
fi
"${PY[@]}" "$CONTEXT/docker/server.py" --config "$SERVE_CONFIG" "${INFER_ARGS[@]}" "${SET_ARGS[@]}" "${WIC[@]}" > /dev/null
check=$("${PY[@]}" "$CONTEXT/docker/server.py" --config "$CONTEXT/configs/LegalMT/serve.image.yaml" --dump-config-field model_dir)
[[ $check == "/models/$MODEL_NAME" ]] || die "image serving config resolves the model to '$check', expected /models/$MODEL_NAME"
note "image serving config: configs/LegalMT/serve.image.yaml (model.local_path=/models/$MODEL_NAME)"

{
    echo "image: $IMAGE"
    echo "staged: $(date -Iseconds) on $(hostname) by $(id -un)"
    echo "repo: $REPO_ROOT"
    echo "serving config: $SERVE_CONFIG ${INFER_ARGS[*]} ${SET_ARGS[*]}"
    echo "inference config: $CONFIG_NAME ($BACKEND backend)"
    echo "model: $SOURCE_DESC -> /models/$MODEL_NAME"
    echo "adapter: ${ADAPTER:-none}"
    echo "licence: $LICENCE ($LIC_SRC)"
    echo "uv.lock sha256: $(sha256sum "$REPO_ROOT/uv.lock" | cut -d' ' -f1)"
} > "$CONTEXT/BUILD_INFO.txt"
note "context size: $(du -sh "$CONTEXT" | cut -f1) (weights hardlinked from the cache when on the same filesystem)"

BUILD_CMD=(docker build -f "$CONTEXT/docker/Dockerfile" -t "$IMAGE" --label "legalmt.build_context=$CONTEXT")
[[ -z $BASE_IMAGE ]] || BUILD_CMD+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
[[ -z $UV_IMAGE ]] || BUILD_CMD+=(--build-arg "UV_IMAGE=$UV_IMAGE")
BUILD_CMD+=("$CONTEXT")

if [[ $STAGE_ONLY == 1 ]]; then
    note "--stage-only: not building. On a Docker host with this context:"
    echo "  DOCKER_BUILDKIT=1 ${BUILD_CMD[*]}"
    exit 0
fi

note "building (the uv sync layer downloads about 8 GB of wheels on a cold cache)"
DOCKER_BUILDKIT=1 "${BUILD_CMD[@]}"
size=$(docker image inspect --format '{{.Size}}' "$IMAGE")
note "built $IMAGE: $(awk -v b="$size" 'BEGIN { printf "%.2f GB", b / 1e9 }')"
note "next: docker/package.sh $IMAGE"
