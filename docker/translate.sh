#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

PROG="translate.sh"

usage() {
    cat <<EOF
usage: $PROG HOST:PORT INPUT_FILE OUTPUT_FILE

Translate one English or Vietnamese document with the LegalMT server at HOST:PORT
(e.g. localhost:8000 or http://10.0.0.5:8000) and write the translation to OUTPUT_FILE.
Environment: LEGALMT_HEALTH_TIMEOUT (900), LEGALMT_HEALTH_POLL (2), LEGALMT_SRC_LANG (auto),
LEGALMT_MAX_TIME (0 = no limit).
EOF
}

die() {
    local rc=$1
    shift
    echo "$PROG: error: $*" >&2
    exit "$rc"
}

if [[ $# -eq 1 && ( $1 == "-h" || $1 == "--help" ) ]]; then
    usage
    exit 0
fi
if [[ $# -ne 3 ]]; then
    usage >&2
    exit 2
fi

ENDPOINT=$1
INPUT=$2
OUTPUT=$3
HEALTH_TIMEOUT=${LEGALMT_HEALTH_TIMEOUT:-900}
HEALTH_POLL=${LEGALMT_HEALTH_POLL:-2}
SRC_LANG=${LEGALMT_SRC_LANG:-auto}
MAX_TIME=${LEGALMT_MAX_TIME:-0}

[[ $HEALTH_TIMEOUT =~ ^[0-9]+$ ]] || die 2 "LEGALMT_HEALTH_TIMEOUT must be a whole number of seconds, got '$HEALTH_TIMEOUT'"
[[ $HEALTH_POLL =~ ^[0-9]+([.][0-9]+)?$ ]] || die 2 "LEGALMT_HEALTH_POLL must be a number of seconds, got '$HEALTH_POLL'"
[[ $MAX_TIME =~ ^[0-9]+$ ]] || die 2 "LEGALMT_MAX_TIME must be a whole number of seconds, got '$MAX_TIME'"
case "$SRC_LANG" in
    auto | en | vi) ;;
    *) die 2 "LEGALMT_SRC_LANG must be auto, en or vi, got '$SRC_LANG'" ;;
esac
command -v curl > /dev/null 2>&1 || die 1 "curl is required but was not found"
[[ -f $INPUT && -r $INPUT ]] || die 2 "input file not found or not readable: $INPUT"
[[ -n $OUTPUT ]] || die 2 "empty output path"
[[ ! -d $OUTPUT ]] || die 2 "output path is a directory: $OUTPUT"
OUT_DIR=$(dirname -- "$OUTPUT")
mkdir -p -- "$OUT_DIR" 2> /dev/null || die 2 "cannot create the output folder: $OUT_DIR"
[[ -w $OUT_DIR && -x $OUT_DIR ]] || die 2 "the output folder is not writable: $OUT_DIR"

BASE=${ENDPOINT%/}
if [[ $BASE != http://* && $BASE != https://* ]]; then
    BASE="http://$BASE"
fi
HOSTPORT=${BASE#*://}
HOSTPORT=${HOSTPORT%%/*}
if [[ $HOSTPORT == \[* ]]; then
    HOST=${HOSTPORT%%]*}
    HOST=${HOST#[}
else
    HOST=${HOSTPORT%%:*}
fi
[[ -n $HOST ]] || die 2 "cannot read a host from '$ENDPOINT' (expected HOST:PORT)"

now() {
    if [[ -n ${EPOCHREALTIME:-} ]]; then
        echo "${EPOCHREALTIME/,/.}"
    else
        date +%s.%N
    fi
}
elapsed() {
    awk -v a="$1" -v b="${2:-$(now)}" 'BEGIN { printf "%.3f", b - a }'
}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/legalmt_translate.XXXXXX")
OUT_TMP=""
cleanup() {
    local rc=$?
    rm -rf "$WORK"
    if [[ -n $OUT_TMP && -e $OUT_TMP ]]; then
        rm -f "$OUT_TMP"
    fi
    exit "$rc"
}
trap cleanup EXIT

T_START=$(now)

waited_msg=0
last_note=0
while true; do
    code=$(curl -sS --noproxy "$HOST" --connect-timeout 5 --max-time 10 -o "$WORK/health" -w '%{http_code}' \
        "$BASE/health" 2> "$WORK/health.err") || code=000
    case "$code" in
        200) break ;;
        500) die 1 "the server reports an error: $(head -c 2000 "$WORK/health")" ;;
    esac
    waited=$(elapsed "$T_START")
    if awk -v w="$waited" -v t="$HEALTH_TIMEOUT" 'BEGIN { exit !(w >= t) }'; then
        die 1 "server at $BASE not ready after ${HEALTH_TIMEOUT} s (last HTTP status $code: $(head -c 300 "$WORK/health.err" | tr '\n' ' '))"
    fi
    if [[ $waited_msg == 0 ]]; then
        echo "$PROG: waiting for the server at $BASE to become ready (timeout ${HEALTH_TIMEOUT} s)" >&2
        waited_msg=1
    elif awk -v w="$waited" -v l="$last_note" 'BEGIN { exit !(w - l >= 30) }'; then
        echo "$PROG: still waiting after ${waited%.*} s (HTTP status $code)" >&2
        last_note=$waited
    fi
    sleep "$HEALTH_POLL"
done
T_READY=$(now)

OUT_TMP=$(mktemp "$OUT_DIR/.$(basename -- "$OUTPUT").part.XXXXXX") || die 2 "cannot create a file in $OUT_DIR"
curl_args=(-sS --noproxy "$HOST" --connect-timeout 10
    -H "Content-Type: text/plain; charset=utf-8" -H "Expect:"
    --data-binary "@$INPUT" -D "$WORK/headers" -o "$OUT_TMP" -w '%{http_code}')
if [[ $MAX_TIME != 0 ]]; then
    curl_args+=(--max-time "$MAX_TIME")
fi
T0=$(now)
rc=0
code=$(curl "${curl_args[@]}" "$BASE/translate?src_lang=$SRC_LANG" 2> "$WORK/curl.err") || rc=$?
T1=$(now)
if [[ $rc != 0 ]]; then
    die 1 "translation request to $BASE failed (curl exit $rc: $(head -c 500 "$WORK/curl.err" | tr '\n' ' '))"
fi
if [[ $code != 200 ]]; then
    die 1 "the server answered HTTP $code: $(head -c 2000 "$OUT_TMP")"
fi
if [[ ! -s $OUT_TMP ]] && grep -q '[^[:space:]]' "$INPUT"; then
    die 1 "the server returned an empty translation for a non-empty document"
fi
chmod "$(printf '%o' $((0666 & ~$(umask))))" "$OUT_TMP"
mv -f -- "$OUT_TMP" "$OUTPUT"
OUT_TMP=""
T_END=$(now)

header() {
    tr -d '\r' < "$WORK/headers" | grep -i "^$1:" | tail -n 1 | cut -d: -f2- | sed 's/^[[:space:]]*//' || true
}
SRC=$(header X-Src-Lang)
TGT=$(header X-Tgt-Lang)
SERVER_SECONDS=$(header X-Seconds)
WORDS=$(header X-Words)
SENTENCES=$(header X-Sentences)
COPIED=$(header X-Source-Copied-Units)
[[ $WORDS =~ ^[0-9]+$ ]] || WORDS=$(wc -w < "$INPUT" | tr -d ' ')
[[ $SENTENCES =~ ^[0-9]+$ ]] || SENTENCES=$(grep -c '[^[:space:]]' "$INPUT" || true)
OUT_WORDS=$(wc -w < "$OUTPUT" | tr -d ' ')
REQUEST_SECONDS=$(elapsed "$T0" "$T1")
WAIT_SECONDS=$(elapsed "$T_START" "$T_READY")
TOTAL_SECONDS=$(elapsed "$T_START" "$T_END")
rate() {
    awk -v n="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.2f", n / s; else print "n/a" }'
}

echo "LegalMT translation done"
echo "  input               : $INPUT"
echo "  output              : $OUTPUT"
echo "  direction           : ${SRC:-?} -> ${TGT:-?} (src_lang=$SRC_LANG)"
echo "  source words        : $WORDS"
echo "  sentences (approx.) : $SENTENCES"
echo "  output words        : $OUT_WORDS"
echo "  server ready wait   : $WAIT_SECONDS s"
echo "  request time        : $REQUEST_SECONDS s (upload + translation + download)"
echo "  server time         : ${SERVER_SECONDS:-?} s (translation inside the server)"
echo "  total run time      : $TOTAL_SECONDS s (this script, including the ready wait)"
echo "  words/second        : $(rate "$WORDS" "$REQUEST_SECONDS") (source words / request time)"
echo "  sentences/second    : $(rate "$SENTENCES" "$REQUEST_SECONDS") (approx. sentences / request time)"
if [[ $COPIED =~ ^[0-9]+$ && $COPIED -gt 0 ]]; then
    echo "$PROG: warning: $COPIED translation unit(s) came back untranslated (source text kept); see the server log" >&2
fi
