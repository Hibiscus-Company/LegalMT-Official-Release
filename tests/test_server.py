"""Tests for the Docker serving path without Docker: server, client script and build staging."""
from __future__ import annotations

import http.client
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from LegalMT.translate import Translator
from LegalMT.translate.core import DocResult, registry_revision

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"
SERVER_PY = DOCKER_DIR / "server.py"
TRANSLATE_SH = DOCKER_DIR / "translate.sh"
BUILD_SH = DOCKER_DIR / "build_image.sh"
PACKAGE_SH = DOCKER_DIR / "package.sh"
LOCAL_TEST_SH = REPO_ROOT / "scripts" / "LegalMT" / "serve_local_test.sh"
OWNED = [DOCKER_DIR / n for n in ("Dockerfile", ".dockerignore", "server.py", "translate.sh", "build_image.sh",
                                  "package.sh")] + [
    REPO_ROOT / "configs" / "LegalMT" / "serve.yaml", LOCAL_TEST_SH, Path(__file__).resolve()]

_spec = importlib.util.spec_from_file_location("legalmt_docker_server", SERVER_PY)
srv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = srv
_spec.loader.exec_module(srv)

HAS_CURL = shutil.which("curl") is not None
needs_curl = pytest.mark.skipif(not HAS_CURL, reason="curl not installed")


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=64, initial_indent="  ", subsequent_indent="  ", break_long_words=False)


def en_doc() -> str:
    lines = ["HARBOUR COUNCIL OF SAMPLETOWN                    REPUBLIC OF EXAMPLELAND",
             "      No. 7/2032/QD-HC                                Sampletown, May 2, 2032", "",
             " " * 30 + "DECISION", "", "  Article 1. Scope", ""]
    lines += _wrap("This decision governs how the council licenses small fishing boats that moor in the old "
                   "harbour. Owners shall renew each licence every two years.") + [""]
    lines += ["  Article 2. Fees", ""]
    lines += _wrap("The yearly fee is 40 units per boat. The council may waive it for rescue boats.") + [""]
    lines += ["  a) the owner's name;", "  b) the length of the boat.", "", " " * 60 + "CHAIRPERSON", "",
              " " * 60 + "Robin Example"]
    return "\n".join(lines) + "\n"


def vi_doc() -> str:
    lines = ["HỘI ĐỒNG CẢNG THỊ TRẤN MẪU                        CỘNG HÒA VÍ DỤ",
             "      Số: 7/2032/QĐ-HĐ                                Thị trấn Mẫu, ngày 2 tháng 5 năm 2032", "",
             " " * 30 + "QUYẾT ĐỊNH", "", "  Điều 1. Phạm vi", ""]
    lines += _wrap("Quyết định này quy định việc hội đồng cấp phép cho thuyền đánh cá nhỏ neo đậu tại cảng cũ. "
                   "Chủ thuyền phải gia hạn giấy phép hai năm một lần.") + [""]
    lines += ["  Điều 2. Lệ phí", ""]
    lines += _wrap("Lệ phí hằng năm là 40 đơn vị cho mỗi thuyền. Hội đồng có thể miễn lệ phí cho thuyền cứu hộ.") + [""]
    lines += ["  a) tên chủ thuyền;", "  b) chiều dài của thuyền.", "", " " * 60 + "CHỦ TỊCH", "",
              " " * 60 + "Trần Văn Mẫu"]
    return "\n".join(lines) + "\n"


def echo_config(name: str = "echo_smoke", overrides: dict | None = None):
    serve_cfg = srv.load_serve_config(srv.DEFAULT_SERVE_CONFIG)
    if overrides:
        serve_cfg["overrides"] = overrides
    cfg, path, _ = srv.infer_config_for(serve_cfg, srv.DEFAULT_SERVE_CONFIG, name)
    return cfg, path


def start(state, host: str = "127.0.0.1", port: int = 0) -> SimpleNamespace:
    httpd = srv.make_server(state, host, port)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    p = httpd.server_address[1]
    return SimpleNamespace(state=state, httpd=httpd, thread=t, port=p, endpoint=f"127.0.0.1:{p}")


def stop(s: SimpleNamespace) -> None:
    s.httpd.shutdown()
    s.httpd.server_close()
    s.state.close()


def request(port: int, method: str, path: str, body=None, headers: dict | None = None, chunked: bool = False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, body=body, headers=headers or {}, encode_chunked=chunked)
        r = conn.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        conn.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_sh(args: list, env: dict | None = None, timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    e = {**os.environ, **(env or {})}
    return subprocess.run(["bash", *map(str, args)], capture_output=True, text=True, env=e, timeout=timeout, **kw)


@pytest.fixture(scope="module")
def echo_server():
    cfg, path = echo_config()
    state = srv.ServeState(cfg, server_opts={"memory_sample_interval": None}, serve_config=srv.DEFAULT_SERVE_CONFIG,
                           infer_config=path)
    state.load()
    s = start(state)
    s.reference = Translator(cfg)
    yield s
    stop(s)


class StubTranslator:
    def __init__(self, delay: float = 0.0, error: Exception | None = None, stats: dict | None = None) -> None:
        self.delay, self.error, self.stats = delay, error, dict(stats or {})
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def translate_text(self, text, src_lang=None, doc_id="text"):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.error:
                raise self.error
            src = src_lang or "en"
            return DocResult(doc_id=doc_id, src_lang=src, tgt_lang="vi" if src == "en" else "en", text=text.upper(),
                             units=[{"src": text}], stats={"n_units": 1, "n_chunks": 1, **self.stats})
        finally:
            with self._lock:
                self.active -= 1

    def info(self):
        return {"backend": "stub"}

    def close(self):
        pass


def test_health(echo_server):
    code, headers, body = request(echo_server.port, "GET", "/health")
    assert code == 200 and headers["content-type"].startswith("application/json")
    h = json.loads(body)
    assert {"status", "model", "config"} <= set(h)
    assert h["status"] == "ok"
    assert h["model"]["backend"] == "echo" and h["model"]["config_name"] == "echo_smoke"
    assert h["config"]["config_name"] == "echo_smoke" and h["config"]["infer_config"].endswith("echo_smoke.yaml")
    assert h["warmup_seconds"] is not None and h["load_seconds"] is not None
    assert set(h["requests"]) >= {"served", "failed", "rejected", "words"}
    code, _, body = request(echo_server.port, "GET", "/")
    assert code == 200 and "POST /translate?src_lang=auto|en|vi" in json.loads(body)["endpoints"]


@pytest.mark.parametrize("make_doc,src,tgt", [(en_doc, "en", "vi"), (vi_doc, "vi", "en")])
def test_translate_detects_direction_and_matches_translator(echo_server, make_doc, src, tgt):
    doc = make_doc()
    code, h, body = request(echo_server.port, "POST", "/translate", doc.encode("utf-8"))
    assert code == 200
    assert h["content-type"] == "text/plain; charset=utf-8"
    ref = echo_server.reference.translate_text(doc)
    assert body.decode("utf-8") == ref.text
    assert body.decode("utf-8").split() == doc.split()
    assert (h["x-src-lang"], h["x-tgt-lang"], h["x-direction"]) == (src, tgt, f"{src}2{tgt}")
    assert int(h["x-words"]) == len(doc.split())
    assert float(h["x-seconds"]) >= 0 and float(h["x-queue-seconds"]) >= 0
    assert int(h["x-units"]) == ref.stats["n_units"] and int(h["x-chunks"]) == ref.stats["n_chunks"]
    assert int(h["x-sentences"]) >= ref.stats["n_units"]


def test_src_lang_parameter(echo_server):
    doc = en_doc().encode("utf-8")
    code, h, _ = request(echo_server.port, "POST", "/translate?src_lang=vi", doc)
    assert code == 200 and (h["x-src-lang"], h["x-tgt-lang"]) == ("vi", "en")
    code, h, _ = request(echo_server.port, "POST", "/translate?src_lang=EN", doc)
    assert code == 200 and h["x-src-lang"] == "en"
    code, _, body = request(echo_server.port, "POST", "/translate?src_lang=fr", doc)
    assert code == 400 and body.startswith(b"error: src_lang must be one of auto|en|vi")


def test_errors_and_edge_cases(echo_server):
    port = echo_server.port
    assert request(port, "GET", "/nope")[0] == 404
    assert request(port, "POST", "/other", b"x")[0] == 404
    code, _, body = request(port, "POST", "/translate", "Điều 1".encode("utf-16"))
    assert code == 400 and b"not valid UTF-8" in body
    code, h, body = request(port, "POST", "/translate", b"")
    assert code == 200 and body == b"" and h["x-words"] == "0"
    code, h, body = request(port, "POST", "/translate", b"  \n\n")
    assert code == 200 and body == b"  \n\n"
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(b"POST /translate HTTP/1.1\r\nHost: test\r\n\r\n")
        assert sock.recv(4096).split(b"\r\n", 1)[0].endswith(b"411 Length Required")
    old = echo_server.state.opts["max_body_bytes"]
    echo_server.state.opts["max_body_bytes"] = 100
    try:
        code, _, body = request(port, "POST", "/translate", b"word " * 100)
        assert code == 413 and b"max_body_bytes" in body
    finally:
        echo_server.state.opts["max_body_bytes"] = old
    assert json.loads(request(port, "GET", "/health")[2])["requests"]["rejected"] >= 4


def test_chunked_upload_and_crlf(echo_server):
    doc = en_doc()
    expected = echo_server.reference.translate_text(doc).text
    parts = [doc[:100].encode("utf-8"), doc[100:].encode("utf-8")]
    code, _, body = request(echo_server.port, "POST", "/translate", iter(parts), {"Transfer-Encoding": "chunked"},
                            chunked=True)
    assert code == 200 and body.decode("utf-8") == expected
    crlf = doc.replace("\n", "\r\n")
    code, _, body = request(echo_server.port, "POST", "/translate", crlf.encode("utf-8"))
    assert code == 200 and body.decode("utf-8") == expected.replace("\n", "\r\n")


def test_lone_cr_mixed_endings_bom_and_nfd(echo_server):
    import unicodedata

    doc = vi_doc()
    expected = echo_server.reference.translate_text(doc).text
    code, h, body = request(echo_server.port, "POST", "/translate", doc.replace("\n", "\r").encode("utf-8"))
    assert code == 200 and body.decode("utf-8") == expected and int(h["x-units"]) > 1
    mixed = doc.replace("\n", "\r\n", 3)
    code, _, body = request(echo_server.port, "POST", "/translate", mixed.encode("utf-8"))
    assert code == 200 and body.decode("utf-8") == expected
    code, h, body = request(echo_server.port, "POST", "/translate", ("﻿" + doc).encode("utf-8"))
    assert code == 200 and h["x-src-lang"] == "vi" and body.decode("utf-8") == expected
    nfd = unicodedata.normalize("NFD", doc)
    assert nfd != doc
    code, h, body = request(echo_server.port, "POST", "/translate", nfd.encode("utf-8"))
    assert code == 200 and h["x-src-lang"] == "vi" and body.decode("utf-8") == expected


def test_keep_alive_serves_two_documents_on_one_connection(echo_server):
    conn = http.client.HTTPConnection("127.0.0.1", echo_server.port, timeout=30)
    try:
        bodies = []
        for make_doc in (en_doc, vi_doc):
            conn.request("POST", "/translate", body=make_doc().encode("utf-8"))
            r = conn.getresponse()
            assert r.status == 200 and r.getheader("Connection") != "close"
            bodies.append(r.read().decode("utf-8"))
    finally:
        conn.close()
    assert bodies == [echo_server.reference.translate_text(d()).text for d in (en_doc, vi_doc)]


def test_request_queued_when_shutdown_starts_gets_503():
    stub = StubTranslator(delay=0.6)
    state = srv.ServeState({"config_name": "stub", "backend": {"name": "echo"}}, translator=stub,
                           server_opts={"memory_sample_interval": None})
    s = start(state)
    try:
        results: dict[str, tuple] = {}

        def call(name: str) -> None:
            results[name] = request(s.port, "POST", "/translate?src_lang=en", name.encode())

        running = threading.Thread(target=call, args=("running",))
        running.start()
        time.sleep(0.2)
        queued = threading.Thread(target=call, args=("queued",))
        queued.start()
        time.sleep(0.2)
        state.status = "stopping"
        running.join()
        queued.join()
        assert results["running"][0] == 200 and results["running"][2] == b"RUNNING"
        assert results["queued"][0] == 503 and b"stopping" in results["queued"][2]
        assert stub.max_active == 1
        counters = state.health()[1]["requests"]
        assert counters["served"] == 1 and counters["failed"] == 0 and counters["rejected"] == 1
    finally:
        stop(s)


def test_fallback_units_reported_in_headers():
    cfg, path = echo_config("echo_smoke", {"backend": {"fail_mode": "merge_lines"}})
    state = srv.ServeState(cfg, server_opts={"memory_sample_interval": None}, infer_config=path)
    state.load(warmup=False)
    s = start(state)
    try:
        code, h, body = request(s.port, "POST", "/translate", en_doc().encode("utf-8"))
        ref = Translator(cfg).translate_text(en_doc())
        assert code == 200 and body.decode("utf-8") == ref.text
        assert int(h["x-fallback-units"]) == ref.stats["fallback_units"] > 0
        assert h["x-source-copied-units"] == "0"
    finally:
        stop(s)


@needs_curl
def test_source_copied_units_are_visible(tmp_path):
    stub = StubTranslator(stats={"fallback_units": 3, "source_copied_units": 2})
    s = start(srv.ServeState({"config_name": "stub", "backend": {"name": "echo"}}, translator=stub,
                             server_opts={"memory_sample_interval": None}))
    try:
        code, h, _ = request(s.port, "POST", "/translate?src_lang=en", b"one line")
        assert code == 200 and h["x-source-copied-units"] == "2" and h["x-fallback-units"] == "3"
        inp, out = tmp_path / "in.md", tmp_path / "out.md"
        inp.write_text("one line", encoding="utf-8")
        r = run_sh([TRANSLATE_SH, s.endpoint, inp, out], env={"TMPDIR": str(tmp_path), "LEGALMT_SRC_LANG": "en"})
        assert r.returncode == 0 and "2 translation unit(s) came back untranslated" in r.stderr
        assert json.loads(request(s.port, "GET", "/health")[2])["requests"]["source_copied_units"] == 4
    finally:
        stop(s)


def test_requests_are_serialised_and_errors_reported():
    stub = StubTranslator(delay=0.3)
    s = start(srv.ServeState({"config_name": "stub", "backend": {"name": "echo"}}, translator=stub,
                             server_opts={"memory_sample_interval": None}))
    try:
        results = []

        def call(k: int) -> None:
            results.append(request(s.port, "POST", "/translate?src_lang=en", f"document {k}".encode()))

        threads = [threading.Thread(target=call, args=(k,)) for k in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.1)
        busy = json.loads(request(s.port, "GET", "/health")[2])
        for t in threads:
            t.join()
        assert busy["busy"] is True and busy["status"] == "ok"
        assert stub.max_active == 1
        assert sorted(r[0] for r in results) == [200, 200, 200]
        assert max(float(r[1]["x-queue-seconds"]) for r in results) >= 0.25
        assert {r[2] for r in results} == {f"DOCUMENT {k}".encode() for k in range(3)}
        stub.error = RuntimeError("engine died")
        code, _, body = request(s.port, "POST", "/translate", b"text")
        assert code == 500 and body.startswith(b"error: translation failed: RuntimeError: engine died")
        h = json.loads(request(s.port, "GET", "/health")[2])
        assert h["status"] == "ok" and h["requests"]["failed"] == 1 and h["requests"]["served"] == 3
    finally:
        stop(s)


def test_approx_sentences_and_newlines():
    f = srv.approx_sentences
    assert f([]) == 0 and f(["", "   "]) == 0
    assert f(["Article 1. Scope"]) == 1
    assert f(["One sentence. Another one. A third"]) == 3
    assert f(["See No. 5 of the list. Then stop."]) == 2
    assert f(["Clause 1.2. The text ends here."]) == 1
    assert f(["a) the name;", "b) the address."]) == 2
    assert f(["Điều 3. Mỗi yêu cầu phải được xem xét. Kết quả được thông báo!"]) == 2
    assert srv.uses_crlf("a\r\nb\r\n") and not srv.uses_crlf("a\r\nb\n") and not srv.uses_crlf("a")
    assert srv.to_lf("a\r\nb\rc\n") == "a\nb\nc\n"


def _approx_sentences_reference(texts) -> int:
    n = 0
    for t in texts:
        t = t.strip()
        if not t:
            continue
        ends, last = 0, -1
        for m in srv._SENT_END.finditer(t):
            before = t[:m.start()].split()
            word = before[-1].strip("(\"'“‘") if before else ""
            if not word or srv._ENUMERATOR.match(word) or word.lower().rstrip(".") in srv._ABBREV:
                continue
            ends, last = ends + 1, m.end()
        n += ends if ends and last == len(t) else ends + 1
    return n


def test_approx_sentences_matches_oracle_and_is_linear():
    cases = [en_doc(), vi_doc(), "Wait ... then go.", " . . ", "“Quoted.” Next one! And (a) more? Yes …",
             "Mr. Sample met Dr. Example. e.g. this. IV. Scope. 2.3. Item", "x.\ty.\nz.", "...", "(1). (b). Word."]
    for c in cases:
        units = [c] + c.splitlines()
        assert srv.approx_sentences(units) == _approx_sentences_reference(units), c
    big = " ".join(["The owner shall pay the yearly fee on time."] * 20000)
    t0 = time.monotonic()
    assert srv.approx_sentences([big]) == 20000
    assert time.monotonic() - t0 < 2.0


def test_repo_serve_config_resolves_without_loading():
    serve_cfg = srv.load_serve_config(srv.DEFAULT_SERVE_CONFIG)
    assert set(serve_cfg["server"]) == set(srv.SERVER_DEFAULTS)
    assert serve_cfg["server"]["port"] == 8000 and serve_cfg["server"]["host"] == "0.0.0.0"
    cfg, path, _ = srv.infer_config_for(serve_cfg, srv.DEFAULT_SERVE_CONFIG)
    assert path == (REPO_ROOT / "configs/LegalMT/infer/hymt15_vllm_greedy.yaml").resolve()
    assert cfg["backend"]["name"] == "vllm" and cfg["backend"]["enforce_eager"] is True
    d = srv.describe(cfg, path, srv.DEFAULT_SERVE_CONFIG)
    assert d["revision"] == registry_revision(d["repo_id"])
    assert d["model_dir"].endswith(f"models--tencent--HY-MT1.5-1.8B/snapshots/{d['revision']}")
    cfg2, _, ov = srv.infer_config_for(serve_cfg, None, None, ["backend.gpu_memory_utilization=0.5"])
    assert cfg2["backend"]["gpu_memory_utilization"] == 0.5 and ov == {"backend": {"gpu_memory_utilization": 0.5}}


def test_serve_config_validation(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("infer_config: echo_smoke\nserver: {prot: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="server.prot"):
        srv.load_serve_config(p)
    p.write_text("server: {port: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="infer_config is required"):
        srv.load_serve_config(p)
    p.write_text("infer_config: echo_smoke\nmodel: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        srv.load_serve_config(p)
    with pytest.raises(ValueError, match="newline"):
        srv.ServeState({}, server_opts={"newline": "crlf"})


def test_hf_snapshot_dir(tmp_path):
    base = tmp_path / "hub" / "models--org--name"
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    assert srv.hf_snapshot_dir("org/name", "main", str(tmp_path)) == base / "snapshots" / "abc123"
    assert srv.hf_snapshot_dir("org/name", "f00d", str(tmp_path)) == base / "snapshots" / "f00d"


def test_write_image_config(tmp_path):
    serve_cfg = srv.load_serve_config(srv.DEFAULT_SERVE_CONFIG)
    out = tmp_path / "serve.image.yaml"
    srv.write_image_config(serve_cfg, srv.DEFAULT_SERVE_CONFIG, out, model_local_path="/models/HY-MT1.5-1.8B",
                           set_items=["backend.max_num_seqs=32"], meta={"licence": "x"})
    img = srv.load_serve_config(out)
    assert img["infer_config"] == "configs/LegalMT/infer/hymt15_vllm_greedy.yaml"
    assert img["image"] == {"licence": "x"}
    cfg, _, _ = srv.infer_config_for(img, out)
    assert cfg["model"]["local_path"] == "/models/HY-MT1.5-1.8B" and cfg["backend"]["max_num_seqs"] == 32
    assert cfg["model"]["repo_id"] == "tencent/HY-MT1.5-1.8B"
    srv.write_image_config(serve_cfg, srv.DEFAULT_SERVE_CONFIG, out, model_local_path="/models/ft",
                           infer_ref="hymt2_hf_greedy", adapter_path="/models/adapter", keep_model_id=False)
    cfg, _, _ = srv.infer_config_for(srv.load_serve_config(out), out)
    assert cfg["model"]["repo_id"] is None and cfg["model"]["revision"] is None
    assert cfg["lora_adapter"] == "/models/adapter" and cfg["backend"]["name"] == "hf"
    outside = tmp_path / "custom.yaml"
    outside.write_text((REPO_ROOT / "configs/LegalMT/infer/hymt15_hf_greedy.yaml").read_text().replace(
        "extends: _base_hymt.yaml", f"extends: {REPO_ROOT / 'configs/LegalMT/infer/_base_hymt.yaml'}"))
    with pytest.raises(ValueError, match="outside"):
        srv.write_image_config(serve_cfg, None, out, model_local_path="/models/x", infer_ref=str(outside))


def test_cli_dump_config_field():
    r = subprocess.run([sys.executable, str(SERVER_PY), "--dump-config-field", "config_name"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "hymt15_vllm_greedy"
    r = subprocess.run([sys.executable, str(SERVER_PY), "--infer-config", "echo_smoke", "--dump-config-field",
                        "model_dir"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == ""
    r = subprocess.run([sys.executable, str(SERVER_PY), "--dump-config"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and json.loads(r.stdout)["backend"] == "vllm"


def test_cli_serves_and_stops_cleanly(tmp_path):
    port_file, pid_file = tmp_path / "port", tmp_path / "pid"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen([sys.executable, str(SERVER_PY), "--infer-config", "echo_smoke", "--host", "127.0.0.1",
                             "--port", "0", "--port-file", str(port_file), "--pid-file", str(pid_file)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        for _ in range(200):
            if port_file.exists() and port_file.read_text().strip():
                break
            assert proc.poll() is None, proc.stdout.read()
            time.sleep(0.05)
        port = int(port_file.read_text())
        assert int(pid_file.read_text()) == proc.pid
        code, h, _ = request(port, "POST", "/translate", vi_doc().encode("utf-8"))
        assert code == 200 and h["x-src-lang"] == "vi"
    finally:
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert "LegalMT.serve: stopped" in out and "request 1: vi2en (auto)" in out


def test_cli_port_clash_fails_fast():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, str(SERVER_PY), "--infer-config", "echo_smoke", "--host", "127.0.0.1",
                            "--port", str(port)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "cannot bind" in r.stderr and time.monotonic() - t0 < 30


def test_offline_env_forced(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    srv.apply_offline_env()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


@needs_curl
def test_translate_sh_roundtrip(echo_server, tmp_path):
    inp, out = tmp_path / "in.md", tmp_path / "sub" / "out.md"
    inp.write_text(en_doc(), encoding="utf-8")
    r = run_sh([TRANSLATE_SH, echo_server.endpoint, inp, out], env={"TMPDIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert out.read_text(encoding="utf-8") == echo_server.reference.translate_text(en_doc()).text
    for key in ("direction           : en -> vi (src_lang=auto)", f"source words        : {len(en_doc().split())}",
                "total run time", "words/second", "sentences/second", "sentences (approx.)"):
        assert key in r.stdout, key
    assert "untranslated" not in r.stderr
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".out.md.part")]
    umask = os.umask(0)
    os.umask(umask)
    assert out.stat().st_mode & 0o777 == 0o666 & ~umask
    r = run_sh([TRANSLATE_SH, f"http://{echo_server.endpoint}/", inp, out],
               env={"TMPDIR": str(tmp_path), "LEGALMT_SRC_LANG": "vi"})
    assert r.returncode == 0 and "direction           : vi -> en (src_lang=vi)" in r.stdout


@needs_curl
def test_translate_sh_waits_for_a_late_server(tmp_path):
    port = free_port()
    inp, out = tmp_path / "in.md", tmp_path / "out.md"
    inp.write_text(vi_doc(), encoding="utf-8")
    env = {**os.environ, "TMPDIR": str(tmp_path), "LEGALMT_HEALTH_POLL": "0.2", "LEGALMT_HEALTH_TIMEOUT": "60"}
    proc = subprocess.Popen(["bash", str(TRANSLATE_SH), f"127.0.0.1:{port}", str(inp), str(out)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    time.sleep(1.0)
    assert proc.poll() is None
    cfg, _ = echo_config()
    state = srv.ServeState(cfg, server_opts={"memory_sample_interval": None})
    state.load(warmup=False)
    s = start(state, port=port)
    try:
        so, se = proc.communicate(timeout=60)
    finally:
        stop(s)
    assert proc.returncode == 0, se
    assert "waiting for the server" in se and "direction           : vi -> en" in so
    assert out.read_text(encoding="utf-8").split() == vi_doc().split()


@needs_curl
def test_translate_sh_failures(echo_server, tmp_path):
    inp, out = tmp_path / "in.md", tmp_path / "out.md"
    inp.write_text(en_doc(), encoding="utf-8")
    env = {"TMPDIR": str(tmp_path)}
    r = run_sh([TRANSLATE_SH, echo_server.endpoint, inp], env=env)
    assert r.returncode == 2 and "usage:" in r.stderr
    assert run_sh([TRANSLATE_SH, "--help"], env=env).returncode == 0
    r = run_sh([TRANSLATE_SH, echo_server.endpoint, tmp_path / "missing.md", out], env=env)
    assert r.returncode == 2 and "input file not found" in r.stderr
    r = run_sh([TRANSLATE_SH, echo_server.endpoint, inp, out], env={**env, "LEGALMT_SRC_LANG": "fr"})
    assert r.returncode == 2 and "LEGALMT_SRC_LANG" in r.stderr
    r = run_sh([TRANSLATE_SH, f"127.0.0.1:{free_port()}", inp, out],
               env={**env, "LEGALMT_HEALTH_TIMEOUT": "1", "LEGALMT_HEALTH_POLL": "0.2"})
    assert r.returncode == 1 and "not ready after 1 s" in r.stderr
    bad = tmp_path / "bad.md"
    bad.write_bytes("Điều 1".encode("utf-16"))
    r = run_sh([TRANSLATE_SH, echo_server.endpoint, bad, out], env=env)
    assert r.returncode == 1 and "HTTP 400" in r.stderr and "not valid UTF-8" in r.stderr
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if ".part." in p.name or p.name.startswith("legalmt_translate.")]


@needs_curl
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_translate_sh_checks_the_output_folder_before_waiting(tmp_path):
    inp = tmp_path / "in.md"
    inp.write_text(en_doc(), encoding="utf-8")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        t0 = time.monotonic()
        r = run_sh([TRANSLATE_SH, f"127.0.0.1:{free_port()}", inp, locked / "out.md"],
                   env={"TMPDIR": str(tmp_path), "LEGALMT_HEALTH_TIMEOUT": "60"})
        assert r.returncode == 2 and "not writable" in r.stderr
        assert time.monotonic() - t0 < 20
    finally:
        locked.chmod(0o700)


def _has_locale(name: str) -> bool:
    try:
        out = subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return name.lower().replace("-", "") in out.lower().replace("-", "")


@needs_curl
@pytest.mark.skipif(not _has_locale("vi_VN.utf8"), reason="vi_VN.UTF-8 locale not installed")
def test_translate_sh_statistics_ignore_a_comma_decimal_locale(echo_server, tmp_path):
    inp, out = tmp_path / "in.md", tmp_path / "out.md"
    inp.write_text(vi_doc(), encoding="utf-8")
    r = run_sh([TRANSLATE_SH, echo_server.endpoint, inp, out], env={"TMPDIR": str(tmp_path), "LC_ALL": "vi_VN.UTF-8"})
    assert r.returncode == 0, r.stderr
    for label, pattern in (("total run time", r"\d+\.\d{3} s"), ("request time", r"\d+\.\d{3} s"),
                           ("words/second", r"(\d+\.\d{2}|n/a)"), ("sentences/second", r"(\d+\.\d{2}|n/a)")):
        line = next(ln for ln in r.stdout.splitlines() if ln.strip().startswith(label))
        assert re.search(r":\s+" + pattern + r"(\s|$)", line), line


def _dockerfile_copy_sources() -> list[str]:
    text = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    srcs = []
    for line in text.splitlines():
        m = re.match(r"^COPY\s+(.*)$", line.strip())
        if m and "--from=" not in m.group(1):
            srcs += m.group(1).split()[:-1]
    return srcs


def _fake_model(root: Path, licence: str) -> Path:
    d = root / "my-merged-model"
    d.mkdir()
    (d / "config.json").write_text('{"architectures": ["HunYuanDenseV1ForCausalLM"]}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * 64)
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (d / "License.txt").write_text(licence, encoding="utf-8")
    (d / "train").mkdir()
    (d / "train" / "notes.txt").write_text("not needed to serve", encoding="utf-8")
    return d


def test_build_image_stage_only(tmp_path):
    model = _fake_model(tmp_path, "TENCENT HY COMMUNITY LICENSE AGREEMENT\n(synthetic stand-in for the test)\n")
    ctx = tmp_path / "ctx"
    env = {"LEGALMT_PYTHON": sys.executable, "TMPDIR": str(tmp_path), "LEGALMT_ARTIFACTS_ROOT": str(tmp_path / "art")}
    r = run_sh([BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "hymt15_hf_greedy", "--model-dir",
                model, "--context", ctx, "--image", "legalmt:test", "--team", "Test Team"], env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "docker build -f" in r.stdout and "legalmt.build_context=" in r.stdout

    srcs = _dockerfile_copy_sources()
    assert {"pyproject.toml", "uv.lock", "models/", "licenses/", "src/", "configs/", "docker/"} <= set(srcs)
    for s in srcs:
        assert (ctx / s).exists(), s
    assert sorted(p.name for p in ctx.iterdir()) == sorted(
        [".dockerignore", "BUILD_INFO.txt", "configs", "docker", "licenses", "models", "pyproject.toml", "src", "uv.lock"])
    assert not list(ctx.rglob("__pycache__")) and not list(ctx.rglob("*.csv")) and not list(ctx.rglob("*.jsonl"))
    assert sorted(p.name for p in (ctx / "configs/LegalMT").iterdir()) == [
        "infer", "models.yaml", "paths.yaml", "serve.image.yaml", "serve.yaml"]
    assert (ctx / ".dockerignore").read_text() == (DOCKER_DIR / ".dockerignore").read_text()
    for name in ("Dockerfile", "server.py", "translate.sh", "build_image.sh", "package.sh"):
        assert (ctx / "docker" / name).is_file()

    staged = ctx / "models" / "my-merged-model"
    assert sorted(p.name for p in staged.iterdir()) == ["License.txt", "config.json", "model.safetensors",
                                                        "tokenizer_config.json"]
    assert (staged / "model.safetensors").stat().st_ino == (model / "model.safetensors").stat().st_ino

    notice = (ctx / "licenses" / "NOTICE").read_text(encoding="utf-8")
    assert "Tencent HY is licensed under the Tencent HY Community License Agreement, Copyright © 2025 Tencent." in notice
    assert "5(a) and 5(b)" in notice and "Test Team" in notice and "Modified files" in notice
    assert "not affiliated with" in notice
    assert (ctx / "licenses" / "License.txt").is_file()

    img = yaml.safe_load((ctx / "configs/LegalMT/serve.image.yaml").read_text(encoding="utf-8"))
    assert img["infer_config"] == "configs/LegalMT/infer/hymt15_hf_greedy.yaml"
    assert img["overrides"]["model"] == {"local_path": "/models/my-merged-model", "repo_id": None, "revision": None}
    assert img["image"]["licence"] == "tencent-hy-community" and img["image"]["image"] == "legalmt:test"

    port_file = tmp_path / "port"
    proc = subprocess.Popen([sys.executable, str(ctx / "docker" / "server.py"), "--config",
                             str(ctx / "configs/LegalMT/serve.image.yaml"), "--infer-config", "echo_smoke",
                             "--host", "127.0.0.1", "--port", "0", "--port-file", str(port_file)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    try:
        for _ in range(200):
            if port_file.exists() and port_file.read_text().strip():
                break
            assert proc.poll() is None, proc.stdout.read()
            time.sleep(0.05)
        port = int(port_file.read_text())
        h = json.loads(request(port, "GET", "/health")[2])
        assert h["config"]["infer_config"].startswith(str(ctx))
        assert h["image"]["model_name"] == "my-merged-model"
        code, hdr, body = request(port, "POST", "/translate", en_doc().encode("utf-8"))
        assert code == 200 and hdr["x-direction"] == "en2vi" and body.decode("utf-8").split() == en_doc().split()
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=30)
    assert proc.returncode == 0

    if HAS_CURL and shutil.which("uv"):
        r = run_sh([LOCAL_TEST_SH, "--echo", "--context", ctx], env={"TMPDIR": str(tmp_path)}, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"context={ctx}" in r.stdout and "PASS: 2 documents" in r.stdout

    r = run_sh([BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "hymt15_hf_greedy", "--model-dir",
                model, "--context", ctx], env=env)
    assert r.returncode == 1 and "--force" in r.stderr


def test_build_image_apache_snapshot_and_refusals(tmp_path):
    model = _fake_model(tmp_path, "Apache License\nVersion 2.0, January 2004\n")
    env = {"LEGALMT_PYTHON": sys.executable, "TMPDIR": str(tmp_path), "LEGALMT_ARTIFACTS_ROOT": str(tmp_path / "art")}
    r = run_sh([BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "hymt2_hf_greedy", "--set",
                f"model.local_path={model}", "--image", "legalmt:apache"], env=env)
    assert r.returncode == 0, r.stderr
    ctx = tmp_path / "art" / "docker_build" / "legalmt_apache"
    notice = (ctx / "licenses" / "NOTICE").read_text(encoding="utf-8")
    assert "Apache License, Version 2.0" in notice and "Modified files" not in notice
    img = yaml.safe_load((ctx / "configs/LegalMT/serve.image.yaml").read_text(encoding="utf-8"))
    assert img["overrides"]["model"]["local_path"] == "/models/Hy-MT2-1.8B"
    assert "repo_id" not in img["overrides"]["model"]
    r = run_sh([BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "echo_smoke"], env=env)
    assert r.returncode == 1 and "echo test backend" in r.stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    r = run_sh([BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "hymt15_hf_greedy", "--model-dir",
                empty], env=env)
    assert r.returncode == 1 and "safetensors" in r.stderr
    assert run_sh([BUILD_SH, "--bogus"], env=env).returncode == 1
    assert run_sh([BUILD_SH, "--help"], env=env).returncode == 0


def test_build_image_merged_model_never_reapplies_the_config_adapter(tmp_path):
    model = _fake_model(tmp_path, "TENCENT HY COMMUNITY LICENSE AGREEMENT\n(synthetic stand-in for the test)\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    env = {"LEGALMT_PYTHON": sys.executable, "TMPDIR": str(tmp_path), "LEGALMT_ARTIFACTS_ROOT": str(tmp_path / "art")}
    base = [BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "hymt15_hf_greedy",
            "--set", f"lora_adapter={adapter}", "--model-dir", model]

    r = run_sh(base + ["--context", tmp_path / "ctx0"], env=env)
    assert r.returncode == 1 and "--no-adapter" in r.stderr and "merged weights" in r.stderr
    assert not (tmp_path / "ctx0").exists()

    r = run_sh(base + ["--no-adapter", "--context", tmp_path / "ctx1"], env=env)
    assert r.returncode == 0, r.stderr
    img = yaml.safe_load((tmp_path / "ctx1/configs/LegalMT/serve.image.yaml").read_text(encoding="utf-8"))
    assert img["overrides"]["lora_adapter"] is None and not (tmp_path / "ctx1/models/adapter").exists()
    cfg, _, _ = srv.infer_config_for(img, tmp_path / "ctx1/configs/LegalMT/serve.image.yaml")
    assert cfg["lora_adapter"] is None
    notice = (tmp_path / "ctx1/licenses/NOTICE").read_text(encoding="utf-8")
    assert "Modified files" in notice and "LoRA adapter" not in notice
    assert str(REPO_ROOT) not in (tmp_path / "ctx1/configs/LegalMT/serve.image.yaml").read_text(encoding="utf-8")

    r = run_sh(base + ["--adapter", adapter, "--context", tmp_path / "ctx2"], env=env)
    assert r.returncode == 0, r.stderr
    img = yaml.safe_load((tmp_path / "ctx2/configs/LegalMT/serve.image.yaml").read_text(encoding="utf-8"))
    assert img["overrides"]["lora_adapter"] == "/models/adapter"
    assert (tmp_path / "ctx2/models/adapter/adapter_config.json").is_file()

    r = run_sh(base + ["--adapter", adapter, "--no-adapter", "--context", tmp_path / "ctx3"], env=env)
    assert r.returncode == 1 and "contradict" in r.stderr

    r = run_sh([BUILD_SH, "--stage-only", "--allow-root-disk", "--infer-config", "hymt2_hf_greedy", "--set",
                f"model.local_path={model}", "--set", f"lora_adapter={adapter}", "--context", tmp_path / "ctx4"], env=env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "ctx4/models/adapter/adapter_config.json").is_file()


@needs_curl
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_serve_local_test_script_echo(tmp_path):
    r = run_sh([LOCAL_TEST_SH, "--echo"], env={"TMPDIR": str(tmp_path)}, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    for key in ("ok  synthetic_en: en -> vi", "ok  synthetic_vi: vi -> en", "server stopped cleanly (exit 0)",
                "PASS: 2 documents", '"status": "ok"'):
        assert key in r.stdout, key
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("serve_local_test.")]
    r = run_sh([LOCAL_TEST_SH, "--bogus"], env={"TMPDIR": str(tmp_path)})
    assert r.returncode == 2


@pytest.mark.skipif(shutil.which("docker") is not None, reason="docker is installed here")
def test_scripts_refuse_without_docker(tmp_path):
    env = {"LEGALMT_PYTHON": sys.executable, "TMPDIR": str(tmp_path), "LEGALMT_ARTIFACTS_ROOT": str(tmp_path)}
    r = run_sh([BUILD_SH], env=env)
    assert r.returncode == 1 and "docker is not installed" in r.stderr and "--stage-only" in r.stderr
    r = run_sh([PACKAGE_SH, "legalmt:test"], env=env)
    assert r.returncode == 1 and "docker is not installed" in r.stderr
    assert run_sh([PACKAGE_SH], env=env).returncode == 2


def test_dockerfile_static():
    text = (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    assert "ARG BASE_IMAGE=python:3.12-slim-bookworm" in lines
    assert any(ln.startswith("FROM ${UV_IMAGE} AS uv") for ln in lines) and "FROM ${BASE_IMAGE}" in lines
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in lines
    assert "uv sync --frozen --group gpu --no-dev" in text
    for env in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "VLLM_USE_FLASHINFER_SAMPLER=0", "VLLM_NO_USAGE_STATS=1",
                "LEGALMT_SERVE_CONFIG=/app/configs/LegalMT/serve.image.yaml", "UV_PROJECT_ENVIRONMENT=/app/.venv"):
        assert env in text, env
    assert "EXPOSE 8000" in lines
    assert 'CMD ["/app/.venv/bin/python", "/app/docker/server.py"]' in lines
    assert 'ENTRYPOINT ["/usr/bin/tini", "--"]' in lines
    assert "ghcr.io/astral-sh/uv:0.11.7" in text
    assert srv.SERVER_DEFAULTS["port"] == 8000 and "127.0.0.1:8000/health" in text


def test_shell_scripts_parse():
    for sh in (TRANSLATE_SH, BUILD_SH, PACKAGE_SH, LOCAL_TEST_SH):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        assert r.returncode == 0, f"{sh}: {r.stderr}"
        assert os.access(sh, os.X_OK), f"{sh} is not executable"
        assert sh.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n")
    if shutil.which("shellcheck"):
        r = subprocess.run(["shellcheck", "-x", *map(str, (TRANSLATE_SH, BUILD_SH, PACKAGE_SH, LOCAL_TEST_SH))],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout


def test_no_long_dashes_in_owned_files():
    for p in OWNED:
        text = p.read_text(encoding="utf-8")
        assert chr(0x2013) not in text and chr(0x2014) not in text, p


def test_real_tokenizer_from_local_path():
    pytest.importorskip("transformers")
    cfg0, _ = echo_config("echo_hytok_smoke")
    snap = srv.hf_snapshot_dir(cfg0["model"]["repo_id"], cfg0["model"]["revision"])
    if not (snap / "tokenizer.json").is_file():
        pytest.skip(f"HY-MT1.5 snapshot not in the HF cache ({snap})")
    cfg, _ = echo_config("echo_hytok_smoke", {"model": {"local_path": str(snap)}})
    state = srv.ServeState(cfg, server_opts={"memory_sample_interval": None})
    state.load()
    s = start(state)
    try:
        h = json.loads(request(s.port, "GET", "/health")[2])
        assert h["model"]["tokenizer"] != "whitespace" and h["model"]["model_path"] == str(snap)
        code, hdr, body = request(s.port, "POST", "/translate", vi_doc().encode("utf-8"))
        assert code == 200 and hdr["x-src-lang"] == "vi"
        assert body.decode("utf-8") == Translator(cfg).translate_text(vi_doc()).text
    finally:
        stop(s)


@pytest.mark.data
def test_dev_mini_documents_through_the_server(echo_server):
    from LegalMT.splits import docs_for

    df = docs_for("dev_mini").head(3)
    for _, row in df.iterrows():
        for col, lang in (("english_markdown", "en"), ("vietnamese_markdown", "vi")):
            doc = row[col]
            code, h, body = request(echo_server.port, "POST", "/translate", doc.encode("utf-8"))
            ref = echo_server.reference.translate_text(doc)
            assert code == 200 and h["x-src-lang"] == lang
            assert body.decode("utf-8") == ref.text
            assert int(h["x-words"]) == len(doc.split())
            assert int(h["x-sentences"]) == _approx_sentences_reference(u["src"] for u in ref.units)
