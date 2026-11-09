#!/usr/bin/env bash
# Package a built image: <name>.tar.gz with its MD5, translate.sh, licenses/ and README.txt.
#
#     docker/package.sh IMAGE [--out-dir DIR] [--name NAME] [--context DIR] [--force] [--allow-root-disk]
#
# Options:
#   --out-dir DIR     output folder (default $LEGALMT_ARTIFACTS_ROOT/submission/<name>/)
#   --name NAME       archive base name (default: the image reference with / : @ replaced by _)
#   --context DIR     build context to take licenses/ from (default: the image's legalmt.build_context label)
#   --force           overwrite an existing archive
#   --allow-root-disk no effect, kept for compatibility
set -euo pipefail
shopt -s inherit_errexit

PROG="package.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() {
    echo "$PROG: error: $*" >&2
    exit 1
}
note() { echo "[$PROG] $*"; }
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

export LEGALMT_ARTIFACTS_ROOT="${LEGALMT_ARTIFACTS_ROOT:-$SCRIPT_DIR/../../legalmt-artifacts}"

IMAGE=""
OUT_DIR=""
NAME=""
CONTEXT=""
FORCE=0
ALLOW_ROOT=1
need_arg() { [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) need_arg "$@"; OUT_DIR=$2; shift 2 ;;
        --name) need_arg "$@"; NAME=$2; shift 2 ;;
        --context) need_arg "$@"; CONTEXT=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        --allow-root-disk) ALLOW_ROOT=1; shift ;;
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) [[ -z $IMAGE ]] || die "only one IMAGE argument is allowed"; IMAGE=$1; shift ;;
    esac
done
[[ -n $IMAGE ]] || { usage >&2; exit 2; }

command -v docker > /dev/null 2>&1 || die "docker is not installed on this machine: package on the Docker host that built the image (see docker/README.md)"
docker image inspect "$IMAGE" > /dev/null 2>&1 || die "no local image '$IMAGE' (build it with docker/build_image.sh)"

NAME=${NAME:-$(printf '%s' "$IMAGE" | tr '/:@' '___')}
OUT_DIR=${OUT_DIR:-$LEGALMT_ARTIFACTS_ROOT/submission/$NAME}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
if [[ $ALLOW_ROOT == 0 ]]; then
    mount_point=$(df -P "$OUT_DIR" | awk 'NR == 2 { print $6 }')
    [[ $mount_point != / ]] || die "$OUT_DIR is on the root filesystem ; use --out-dir on a larger filesystem"
fi
ARCHIVE="$OUT_DIR/$NAME.tar.gz"
if [[ -e $ARCHIVE && $FORCE == 0 ]]; then
    die "$ARCHIVE exists; pass --force to overwrite"
fi

size=$(docker image inspect --format '{{.Size}}' "$IMAGE")
free=$(df -P -B1 "$OUT_DIR" | awk 'NR == 2 { print $4 }')
need=$size
root_dir=$(docker info --format '{{.DockerRootDir}}' 2> /dev/null || true)
if [[ -n $root_dir && -d $root_dir ]] && [[ $(df -P "$root_dir" | awk 'NR == 2 { print $6 }') == $(df -P "$OUT_DIR" | awk 'NR == 2 { print $6 }') ]]; then
    need=$((size * 2))
fi
[[ $free -gt $need ]] || die "$OUT_DIR has $(awk -v b="$free" 'BEGIN { printf "%.1f", b / 1e9 }') GB free, needs $(awk -v b="$need" 'BEGIN { printf "%.1f", b / 1e9 }') GB (image $(awk -v b="$size" 'BEGIN { printf "%.1f", b / 1e9 }') GB, staged by dockerd plus the archive)"

if command -v pigz > /dev/null 2>&1; then
    GZ=(pigz -6)
else
    GZ=(gzip -6)
fi
note "saving $IMAGE ($(awk -v b="$size" 'BEGIN { printf "%.2f", b / 1e9 }') GB) to $ARCHIVE with ${GZ[0]}"
t0=$(date +%s)
docker save "$IMAGE" | "${GZ[@]}" > "$ARCHIVE.part"
mv -f "$ARCHIVE.part" "$ARCHIVE"
(cd "$OUT_DIR" && md5sum "$NAME.tar.gz" > "$NAME.tar.gz.md5")
note "archive $(du -h "$ARCHIVE" | cut -f1) in $(($(date +%s) - t0)) s, md5 $(cut -d' ' -f1 "$ARCHIVE.md5")"

cp "$SCRIPT_DIR/translate.sh" "$OUT_DIR/translate.sh"
chmod +x "$OUT_DIR/translate.sh"

CONTEXT=${CONTEXT:-$(docker image inspect --format '{{ index .Config.Labels "legalmt.build_context" }}' "$IMAGE" 2> /dev/null || true)}
if [[ -n $CONTEXT && -d $CONTEXT/licenses ]]; then
    rm -rf "$OUT_DIR/licenses"
    cp -R "$CONTEXT/licenses" "$OUT_DIR/licenses"
    note "licences copied from $CONTEXT/licenses"
else
    echo "$PROG: WARNING: no build context with licenses/ found (pass --context): the model licence and NOTICE must accompany the upload" >&2
fi

cat > "$OUT_DIR/README.txt" << EOF
LegalMT: VLSP 2026 Document-Level Legal Machine Translation submission
=====================================================================

Files
  $NAME.tar.gz        Docker image $IMAGE (docker save, gzip)
  $NAME.tar.gz.md5    MD5 checksum
  translate.sh        client script: translate.sh HOST:PORT INPUT_FILE OUTPUT_FILE
  licenses/           model licence and NOTICE (also in the image under /licenses)

Requirements
  An NVIDIA GPU with a driver that supports CUDA 13 (driver 580 or newer), Docker with the NVIDIA
  Container Toolkit, curl on the host. The image works offline.

1. Check and load
     md5sum -c $NAME.tar.gz.md5
     docker load -i $NAME.tar.gz

2. Start the server (it listens on port 8000 inside the container)
     docker run -d --name legalmt --gpus all --shm-size 2g -p 8000:8000 $IMAGE
   Loading the model takes about a minute; "docker logs legalmt" shows "ready on ..." when done.

3. Translate one document (English or Vietnamese; the direction is detected automatically)
     ./translate.sh localhost:8000 input.md output.md
   The script waits for the server to be ready, writes the translation to output.md and prints
   the total run time, words per second and (approximate) sentences per second.

4. Stop
     docker stop legalmt
EOF
note "done: $OUT_DIR"
ls -la "$OUT_DIR"
