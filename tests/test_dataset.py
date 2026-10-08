import json

from scanner.cli import main
from scanner.exporters.dataset import build_rows, export_dataset
from scanner.models import StageResult
from tests.conftest import make_paper

DOMAIN_ANSWER = {"selected": "Robotics", "probabilities": {"Robotics": 0.7, "Vision": 0.3, "Other": 0.0}, "confidence": 0.7}


def classify(db, paper_id, probability, decision, with_domain=False):
    db.add_new_papers([make_paper(paper_id, title=f"Paper {paper_id}")])
    results = [StageResult("relevance", "noul", {"probability": probability}, decision, 100, 0, 0.0, "jev")]
    if with_domain:
        results.append(StageResult("domain", "choice", DOMAIN_ANSWER, "accept", 100, 0, 0.0, "jev"))
    status = {"accept": "accepted", "maybe": "maybe", "reject": "rejected"}[decision]
    db.save_classification(paper_id, results, status, probability, {})


def by_id(rows):
    return {row["id"]: row | {key: json.loads(row[key]) for key in ("state", "questions", "gold", "jev")} for row in rows}


def test_labels_keep_their_split_when_changed(db):
    classify(db, "1", 0.5, "maybe")
    db.set_label("1", True, "test")
    db.set_label("1", False)

    assert db.get_labels()["1"]["relevant"] is False
    assert db.get_labels()["1"]["split"] == "test"

    db.remove_label("1")
    assert db.get_labels() == {}


def test_label_counts_and_maybe_queue(db):
    classify(db, "1", 0.5, "maybe")
    classify(db, "2", 0.7, "maybe")
    classify(db, "3", 0.1, "reject")

    assert db.next_unlabelled_maybe()["paper_id"] == "2"  # most likely relevant first
    assert db.next_unlabelled_maybe(skip=["2"])["paper_id"] == "1"

    db.set_label("2", True)
    db.set_label("3", False, "test")
    counts = db.label_counts()

    assert counts["maybe_left"] == 1
    assert counts["train"] == 1
    assert counts["test"] == 1
    assert counts["test_by_status"]["rejected"] == 1


def test_test_sample_draws_from_the_status_with_fewest_labels(db):
    classify(db, "1", 0.9, "accept", with_domain=True)
    classify(db, "2", 0.1, "reject")
    classify(db, "3", 0.05, "reject")
    db.set_label("2", False, "test")

    # Rejected already has a test label; accepted and maybe have none, and no maybe exists.
    assert db.next_test_candidate()["paper_id"] == "1"

    db.set_label("1", True, "test")
    assert db.next_test_candidate()["paper_id"] == "3"
    assert db.next_test_candidate(skip=["3"]) is None


def test_gold_comes_from_jev_and_unlabelled_maybe_is_skipped(db, profile):
    classify(db, "1", 0.9, "accept", with_domain=True)
    classify(db, "2", 0.1, "reject")
    classify(db, "3", 0.5, "maybe")

    rows, summary = build_rows(db, profile)
    rows = by_id(rows)

    assert set(rows) == {"1", "2"}
    assert summary.skipped_maybe == 1
    assert rows["1"]["gold"] == {
        "relevance": True,
        "domain": {"type": "choice", "label": "Robotics", "probabilities": DOMAIN_ANSWER["probabilities"]},
    }
    assert rows["1"]["questions"]["relevance"]["type"] == "noul"
    assert rows["1"]["questions"]["domain"]["criteria"] == {"Robotics": None, "Vision": None, "Other": None}
    assert rows["1"]["state"] == {"title": "Paper 1", "abstract": "Abstract of Paper 1."}
    assert rows["1"]["label_source"] == "jev"
    assert rows["2"]["gold"] == {"relevance": False}
    assert rows["2"]["jev"] == {"relevance": {"probability": 0.1}}


def test_your_label_overrides_jev(db, profile):
    classify(db, "1", 0.9, "accept", with_domain=True)  # Jev says yes, you say no
    classify(db, "2", 0.5, "maybe")                     # Jev unsure, you say yes
    classify(db, "3", 0.1, "reject")                    # held out for testing
    db.set_label("1", False)
    db.set_label("2", True)
    db.set_label("3", True, "test")

    rows = by_id(build_rows(db, profile)[0])

    # Later questions assume an on-topic paper, so they are dropped for a no.
    assert rows["1"]["gold"] == {"relevance": False}
    assert rows["1"]["label_source"] == "human"
    assert rows["2"]["gold"] == {"relevance": True}  # no domain answer from Jev
    assert rows["3"]["split"] == "test"
    assert rows["3"]["gold"] == {"relevance": True}


def test_export_writes_jsonl_files_and_manifest(db, profile, tmp_path):
    classify(db, "1", 0.9, "accept", with_domain=True)
    classify(db, "2", 0.1, "reject")
    db.set_label("2", False, "test")

    summary = export_dataset(db, profile, tmp_path / "out")

    train = (tmp_path / "out" / "train.jsonl").read_text().splitlines()
    test = (tmp_path / "out" / "test.jsonl").read_text().splitlines()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [json.loads(line)["id"] for line in train] == ["1"]
    assert [json.loads(line)["id"] for line in test] == ["2"]
    assert isinstance(json.loads(train[0])["gold"], str)  # JSON strings, as Unsloth expects
    assert summary.rows == {"train": 1, "test": 1}
    assert manifest["files"]["train.jsonl"] == {"rows": 1, "relevant": 1, "not_relevant": 0}
    assert manifest["questions"] == {"relevance": 2, "domain": 1}


def test_cli_export_dataset(tmp_path):
    config = tmp_path / "profile.yaml"
    config.write_text(
        f"topic: Test\ndata_dir: {tmp_path / 'data'}\nsource:\n  categories: [cs.CV]\n"
        "stages:\n  - name: relevance\n    type: noul\n    instructions: Relevant?\n"
    )

    assert main(["export-dataset", "--config", str(config)]) == 0
    assert (tmp_path / "data" / "dataset" / "train.jsonl").exists()
