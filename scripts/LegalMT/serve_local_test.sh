#!/usr/bin/env bash
# Start docker/server.py locally, run docker/translate.sh on synthetic documents, check the outputs, stop the server.
#
#     scripts/LegalMT/serve_local_test.sh --echo
#     scripts/LegalMT/serve_local_test.sh [--infer-config NAME] [--set KEY=VALUE ...]
#     scripts/LegalMT/serve_local_test.sh --hide-gpu --infer-config echo_hytok_smoke
#     scripts/LegalMT/serve_local_test.sh --context <build context dir> [--hide-gpu ...]
#
# Options:
#   --echo                 CPU .venv with the echo backend (no model)
#   --hide-gpu             GPU env with CUDA hidden
#   --infer-config NAME    server --infer-config
#   --serve-config PATH    server --config
#   --context DIR          serve a context staged by docker/build_image.sh
#   --set KEY=VALUE        server --set (repeatable)
#   --input FILE           also translate this document (repeatable)
#   --identity | --no-identity
#                          compare with the layout parser's identity render, or check the target language
#   --health-timeout S     how long translate.sh waits for the server (default 900)
#   --keep                 keep the temporary folder
#
# Prints PASS (exit 0) or FAIL (exit 1).
set -euo pipefail
shopt -s inherit_errexit

PROG="serve_local_test.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="gpu"
INFER=""
SERVE_CONFIG=""
CONTEXT=""
SETS=()
INPUTS=()
IDENTITY="auto"
HEALTH_TIMEOUT=900
KEEP=0

die() {
    echo "$PROG: error: $*" >&2
    exit 2
}
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
need_arg() { [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --echo) MODE="echo"; shift ;;
        --hide-gpu) MODE="hidden"; shift ;;
        --infer-config) need_arg "$@"; INFER=$2; shift 2 ;;
        --serve-config) need_arg "$@"; SERVE_CONFIG=$2; shift 2 ;;
        --context) need_arg "$@"; [[ -f $2/docker/server.py ]] || die "--context $2 has no docker/server.py"; CONTEXT=$(cd "$2" && pwd); shift 2 ;;
        --set) need_arg "$@"; SETS+=(--set "$2"); shift 2 ;;
        --input) need_arg "$@"; [[ -f $2 ]] || die "--input $2 not found"; INPUTS+=("$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2 ;;
        --identity) IDENTITY=1; shift ;;
        --no-identity) IDENTITY=0; shift ;;
        --health-timeout) need_arg "$@"; HEALTH_TIMEOUT=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

SERVER_PY="$REPO_ROOT/docker/server.py"
if [[ -n $CONTEXT ]]; then
    SERVER_PY="$CONTEXT/docker/server.py"
    SERVE_CONFIG=${SERVE_CONFIG:-$CONTEXT/configs/LegalMT/serve.image.yaml}
fi
if [[ $MODE == echo ]]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
    export TMPDIR="${TMPDIR:-/tmp}"
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export CUDA_VISIBLE_DEVICES=""
    INFER=${INFER:-echo_smoke}
    PY=(uv run --project "$REPO_ROOT" --quiet python)
    SERVER_CMD=("${PY[@]}" "$SERVER_PY")
else
    source "$SCRIPT_DIR/env.sh"
    PY=(uv run --project "$REPO_ROOT" --group gpu --quiet python)
    if [[ $MODE == hidden ]]; then
        export CUDA_VISIBLE_DEVICES=""
        SERVER_CMD=("${PY[@]}" "$SERVER_PY")
    else
        SERVER_CMD=("$SCRIPT_DIR/gpu_run.sh" "$SERVER_PY")
    fi
fi
if [[ $IDENTITY == auto ]]; then
    IDENTITY=0
    [[ $(basename "${INFER%.yaml}") == echo* ]] && IDENTITY=1
fi
SERVER_ARGS=()
[[ -z $SERVE_CONFIG ]] || SERVER_ARGS+=(--config "$SERVE_CONFIG")
[[ -z $INFER ]] || SERVER_ARGS+=(--infer-config "$INFER")
if [[ -n $CONTEXT ]]; then
    for key in local_path lora_adapter; do
        val=$("${PY[@]}" "$SERVER_PY" "${SERVER_ARGS[@]}" "${SETS[@]}" --dump-config-field "$key")
        if [[ $val == /models/* ]]; then
            [[ $key == local_path ]] && SETS+=(--set "model.local_path=$CONTEXT$val") || SETS+=(--set "lora_adapter=$CONTEXT$val")
        fi
    done
fi
SERVER_ARGS+=("${SETS[@]}")

mkdir -p "${TMPDIR:-/tmp}"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/serve_local_test.XXXXXX")
BG_PID=""
FAILED=0

server_pid() { cat "$WORK/server.pid" 2> /dev/null || true; }
stop_server() {
    local pid
    pid=$(server_pid)
    if [[ -n $pid ]] && kill -0 "$pid" 2> /dev/null; then
        kill -TERM "$pid" 2> /dev/null || true
    fi
    if [[ -n $BG_PID ]]; then
        for _ in $(seq 1 180); do
            kill -0 "$BG_PID" 2> /dev/null || break
            sleep 0.5
        done
        if kill -0 "$BG_PID" 2> /dev/null; then
            echo "$PROG: server did not stop within 90 s, killing it" >&2
            [[ -z $pid ]] || kill -KILL "$pid" 2> /dev/null || true
            kill -KILL "$BG_PID" 2> /dev/null || true
        fi
    fi
}
cleanup() {
    local rc=$?
    if [[ -n $BG_PID ]] && kill -0 "$BG_PID" 2> /dev/null; then
        stop_server
    fi
    if [[ $rc != 0 || $FAILED != 0 ]]; then
        if [[ -f $WORK/server.log ]]; then
            echo "---- server log (last 40 lines) ----" >&2
            tail -n 40 "$WORK/server.log" >&2 || true
        fi
        echo "$PROG: kept $WORK for inspection" >&2
    elif [[ $KEEP == 1 ]]; then
        echo "[$PROG] kept $WORK"
    else
        rm -rf "$WORK"
    fi
    exit "$rc"
}
trap cleanup EXIT
fail() {
    FAILED=1
    echo "$PROG: FAIL: $*" >&2
    exit 1
}

cat > "$WORK/synthetic_en.md" << 'EOF'
CITY BOARD OF TESTVILLE                         REPUBLIC OF EXAMPLELAND
        -------                                Order - Progress - Harmony
      No. 12/2031/QD-TB                                   Testville, March 3, 2031

                                    REGULATION
                 On the upkeep of public gardens and small parks

  Article 1. Scope

  This regulation sets out how the city board receives, reviews and
  answers written requests from residents about public gardens.

  Article 2. Content of a request

  Each request shall be reviewed within 15 working days from the date of
  receipt. The applicant shall be informed of the outcome in writing.
  a) the name and address of the applicant;
  b) a short description of the garden.

                                                                  ON BEHALF OF THE BOARD
                                                                        CHAIRPERSON


                                                                        Alex Sample
EOF
cat > "$WORK/synthetic_vi.md" << 'EOF'
ỦY BAN THÀNH PHỐ THỬ NGHIỆM                    CỘNG HÒA XÃ HỘI VÍ DỤ
        -------                                Trật tự - Tiến bộ - Hài hòa
      Số: 12/2031/QĐ-TN                                   Thử Nghiệm, ngày 3 tháng 3 năm 2031

                                    QUY CHẾ
                 Về việc chăm sóc vườn hoa và công viên nhỏ

  Điều 1. Phạm vi

  Quy chế này quy định việc tiếp nhận, xem xét và trả lời các yêu cầu
  bằng văn bản của người dân về vườn hoa công cộng.

  Điều 2. Nội dung yêu cầu

  Mỗi yêu cầu phải được xem xét trong thời hạn 15 ngày làm việc kể từ
  ngày nhận. Người nộp đơn phải được thông báo kết quả bằng văn bản.
  a) tên và địa chỉ của người nộp đơn;
  b) mô tả ngắn gọn về khu vườn.

                                                                  TM. ỦY BAN
                                                                  CHỦ TỊCH


                                                                  Nguyễn Văn Mẫu
EOF

check_output() {
    "${PY[@]}" - "$REPO_ROOT/src" "$@" << 'PYCHECK'
import sys
sys.path.insert(0, sys.argv[1])
from LegalMT.langid import detect_lang, other_lang
from LegalMT.layout import parse_document, render_document, translation_units

src_path, out_path, src_lang, identity = sys.argv[2:6]
src = open(src_path, encoding="utf-8", newline="").read()
out = open(out_path, encoding="utf-8", newline="").read()
problems = []
if src.strip() and not out.strip():
    problems.append("empty output")
if identity == "1":
    # echo returns every unit unchanged, so the served document must equal the layout parser's
    # identity render (NOT the raw source: the render reflows wrapped lines into one line)
    # (the server reads CRLF and lone CR as LF before translating: do the same here)
    skel = parse_document(src.replace("\r\n", "\n").replace("\r", "\n"), src_lang)
    expected = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
    if out.replace("\r\n", "\n") != expected:
        problems.append("echo output differs from the layout parser's identity render")
elif detect_lang(out) != other_lang(src_lang):
    problems.append(f"output language {detect_lang(out)}, expected {other_lang(src_lang)}")
print(f"words {len(src.split())} -> {len(out.split())}, lines {src.count(chr(10))} -> {out.count(chr(10))}"
      + ("" if not problems else "; PROBLEM: " + "; ".join(problems)))
sys.exit(1 if problems else 0)
PYCHECK
}

PORT=$("${PY[@]}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
echo "[$PROG] mode=$MODE infer_config=${INFER:-<serve config>} context=${CONTEXT:-<repo>} port=$PORT work=$WORK"
T0=$(date +%s)
"${SERVER_CMD[@]}" --host 127.0.0.1 --port "$PORT" --pid-file "$WORK/server.pid" --port-file "$WORK/server.port" \
    "${SERVER_ARGS[@]}" > "$WORK/server.log" 2>&1 &
BG_PID=$!

run_one() {
    local name=$1 in=$2 expect=$3 out="$WORK/out_$1" log="$WORK/client_$1.log" cpid src tgt
    LEGALMT_HEALTH_TIMEOUT=$HEALTH_TIMEOUT "$REPO_ROOT/docker/translate.sh" "127.0.0.1:$PORT" "$in" "$out" > "$log" 2>&1 &
    cpid=$!
    while kill -0 "$cpid" 2> /dev/null; do
        if ! kill -0 "$BG_PID" 2> /dev/null; then
            kill "$cpid" 2> /dev/null || true
            wait "$cpid" 2> /dev/null || true
            fail "the server exited while translating $name"
        fi
        sleep 0.5
    done
    if ! wait "$cpid"; then
        cat "$log" >&2
        fail "translate.sh failed on $name"
    fi
    read -r src tgt < <(awk '/^  direction/ { print $3, $5 }' "$log") || true
    [[ $expect == any || $src == "$expect" ]] || fail "$name: server detected '$src', expected '$expect'"
    local counts
    counts=$(check_output "$in" "$out" "$src" "$IDENTITY") || { echo "$counts" >&2; fail "$name: output check failed"; }
    local secs wps
    secs=$(awk '/^  request time/ { print $4 }' "$log")
    wps=$(awk '/^  words\/second/ { print $3 }' "$log")
    echo "[$PROG] ok  $name: $src -> $tgt, $counts, request ${secs} s, ${wps} words/s"
}

run_one synthetic_en "$WORK/synthetic_en.md" en
echo "[$PROG] server ready after $(($(date +%s) - T0)) s: $(grep -m1 'ready on' "$WORK/server.log" | sed 's/.*INFO LegalMT.serve: //')"
run_one synthetic_vi "$WORK/synthetic_vi.md" vi
k=0
for f in "${INPUTS[@]}"; do
    k=$((k + 1))
    run_one "input$k" "$f" any
done

curl -sS --noproxy 127.0.0.1 "http://127.0.0.1:$PORT/health" > "$WORK/health.json" || fail "/health did not answer after the run"
"${PY[@]}" -c 'import json, sys
h = json.load(open(sys.argv[1], encoding="utf-8"))
m = h.get("model") or {}
print("[serve_local_test.sh] health: " + json.dumps({"status": h.get("status"), "config": m.get("config_name"),
      "backend": m.get("backend"), "load_seconds": h.get("load_seconds"), "warmup_seconds": h.get("warmup_seconds"),
      "requests": h.get("requests"), "memory": h.get("memory")}))
sys.exit(0 if h.get("status") == "ok" else 1)' "$WORK/health.json" || fail "/health status is not ok"
grep 'LegalMT.serve: request ' "$WORK/server.log" | sed 's/^.*LegalMT.serve: /[server] /' || true

stop_server
rc=0
wait "$BG_PID" || rc=$?
BG_PID=""
[[ $rc == 0 ]] || fail "the server exited with code $rc after SIGTERM (expected a clean 0)"
grep -q 'LegalMT.serve: stopped' "$WORK/server.log" || fail "no clean 'stopped' line in the server log"
echo "[$PROG] server stopped cleanly (exit 0)"
echo "[$PROG] PASS: $((2 + ${#INPUTS[@]})) documents"
