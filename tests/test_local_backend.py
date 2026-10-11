"""The local backend against a fake llama.cpp server over real HTTP."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scanner.evaluate import evaluate
from scanner.exporters.dataset import export_dataset
from scanner.jev_client import JevClient, backend_name
from scanner.models import StageResult
from tests.conftest import RELEVANCE, make_paper


class FakeLlamaServer(BaseHTTPRequestHandler):
    """Answers like llama-server's /v1/systemone: relevant if 'event' is in the title."""

    requests: list = []

    def do_GET(self):
        self._reply(200, {"status": "ok"} if self.path == "/health" else {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlamaServer.requests.append(body)
        relevant = "event" in body["state"]["title"].lower()
        answers = {}
        for name, question in body["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.95 if relevant else 0.03}
            else:
                options = list(question["criteria"])
                probabilities = {option: (0.9 if i == 0 else 0.1 / (len(options) - 1)) for i, option in enumerate(options)}
                answers[name] = {"type": "choice", "choice": options[0], "probabilities": probabilities, "confidence": 0.8}
        self._reply(200, {"model": "event-cameras-q8_0.gguf", "answers": answers, "usage": {"input_tokens": 250, "output_tokens": 0}})

    def _reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def llama_server(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeLlamaServer.requests = []
    monkeypatch.setenv("LOCAL_DECISION_URL", f"http://127.0.0.1:{server.server_port}")
    yield FakeLlamaServer
    server.shutdown()


def test_local_backend_answers_for_free(llama_server):
    client = JevClient(backend="local")

    answer = client.ask(RELEVANCE, {"title": "Event-based optical flow", "abstract": "DVS."})

    assert answer.result == {"probability": 0.95}
    assert answer.model == "event-cameras-q8_0.gguf"
    assert answer.input_tokens == 250
    assert answer.cost_usd == 0.0
    client.close()


def test_backend_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("DECISION_BACKEND", raising=False)
    assert backend_name() == "jev"
    monkeypatch.setenv("DECISION_BACKEND", "Local ")
    assert backend_name() == "local"
    assert backend_name("jev") == "jev"
    monkeypatch.setenv("DECISION_BACKEND", "gpt")
    with pytest.raises(RuntimeError, match="Unknown decision backend"):
        backend_name()


def test_a_stopped_server_fails_early_with_instructions(monkeypatch):
    monkeypatch.setenv("LOCAL_DECISION_URL", "http://127.0.0.1:9")  # nothing listens here
    with pytest.raises(RuntimeError, match="llama-server"):
        JevClient(backend="local")


def classify(db, paper_id, title, decision):
    db.add_new_papers([make_paper(paper_id, title=title)])
    probability = 0.9 if decision == "accept" else 0.1
    results = [StageResult("relevance", "noul", {"probability": probability}, decision, 1, 0, 0.0, "jev")]
    if decision == "accept":
        domain = {"selected": "Robotics", "probabilities": {"Robotics": 0.8, "Vision": 0.1, "Other": 0.1}, "confidence": 0.7}
        results.append(StageResult("domain", "choice", domain, "accept", 1, 0, 0.0, "jev"))
    db.save_classification(paper_id, results, "accepted" if decision == "accept" else "rejected", probability, {})


def test_evaluate_asks_relevance_alone_like_a_scan(llama_server, db, profile, tmp_path):
    classify(db, "1", "Event camera SLAM", "accept")
    classify(db, "2", "Event-based tracking", "accept")
    classify(db, "3", "Graph neural networks", "reject")
    classify(db, "4", "Event detection in news", "reject")  # fooled by the word "event"
    for paper_id, relevant in (("1", True), ("2", True), ("3", False), ("4", False)):
        db.set_label(paper_id, relevant, "test")
    export_dataset(db, profile, tmp_path / "ds")

    report = evaluate(profile, JevClient(backend="local"), tmp_path / "ds" / "test.jsonl")

    # Every request holds exactly one question, so a model cannot read relevance from their number.
    assert [len(request["questions"]) for request in llama_server.requests] == [1] * 6  # 4 relevance + 2 domain
    scores = report.scores()
    assert scores["found"] == 1.0
    assert scores["precision"] == 2 / 3
    assert scores["sub_topics"] == 1.0
    assert [p.paper_id for p in report.false_accepts()] == ["4"]
    assert report.cost_usd == 0.0


def test_cli_evaluate_prints_a_table(llama_server, tmp_path, capsys):
    from scanner.cli import main
    from scanner.config import load_profile
    from scanner.db import Database

    config = tmp_path / "profile.yaml"
    config.write_text(
        f"topic: Test\ndata_dir: {tmp_path / 'data'}\nsource:\n  categories: [cs.CV]\n"
        "stages:\n  - name: relevance\n    type: noul\n    instructions: About event cameras?\n"
    )
    db = Database(load_profile(config).db_path)
    classify(db, "1", "Event camera SLAM", "accept")
    classify(db, "2", "Graph neural networks", "reject")
    db.set_label("1", True, "test")
    db.set_label("2", False, "test")
    export_dataset(db, load_profile(config))
    db.close()

    assert main(["evaluate", "--backend", "local", "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "found" in out and "100.0%" in out
