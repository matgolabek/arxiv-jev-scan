import json

from scanner.cli import main
from scanner.exporters.dataset import build_rows, export_dataset, push_dataset, read_labels_file, write_labels_file
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


def test_labels_round_trip_through_a_file(db, tmp_path):
    from scanner.db import Database

    classify(db, "1", 0.5, "maybe")
    classify(db, "2", 0.1, "reject")
    db.set_label("1", True)
    db.set_label("2", False, "test")
    assert write_labels_file(db, tmp_path / "labels.jsonl") == 2

    fresh = Database(tmp_path / "fresh.db")  # e.g. a new machine, nothing scanned yet
    counts = fresh.import_labels(read_labels_file(tmp_path / "labels.jsonl"))

    assert counts == {"added": 2, "updated": 0, "unchanged": 0}
    assert fresh.get_labels() == db.get_labels()
    assert read_labels_file(tmp_path / "labels.jsonl")[0]["title"] == "Paper 1"
    fresh.close()


def test_import_keeps_the_newer_label(db):
    classify(db, "1", 0.5, "maybe")
    db.set_label("1", True)
    newer = db.get_labels()["1"]["labelled_at"]

    older = [{"paper_id": "1", "relevant": False, "split": "train", "labelled_at": "2000-01-01T00:00:00+00:00"}]
    assert db.import_labels(older) == {"added": 0, "updated": 0, "unchanged": 1}
    assert db.get_labels()["1"]["relevant"] is True

    later = [{"paper_id": "1", "relevant": False, "split": "test", "labelled_at": "2999-01-01T00:00:00+00:00"}]
    assert db.import_labels(later)["updated"] == 1
    assert db.get_labels()["1"] == {"relevant": False, "split": "test", "labelled_at": "2999-01-01T00:00:00+00:00"}
    assert newer < "2999"


def test_imported_labels_for_unscanned_papers_apply_after_the_scan(db, profile):
    db.import_labels([{"paper_id": "1", "relevant": False, "split": "test", "labelled_at": "2026-10-01T00:00:00+00:00"}])
    assert build_rows(db, profile)[0] == []

    classify(db, "1", 0.9, "accept", with_domain=True)

    row = by_id(build_rows(db, profile)[0])["1"]
    assert row["split"] == "test"
    assert row["gold"] == {"relevance": False}


def test_export_writes_a_dataset_card_and_labels_backup(db, profile, tmp_path):
    classify(db, "1", 0.9, "accept", with_domain=True)
    db.set_label("1", True)

    export_dataset(db, profile, tmp_path / "out")

    card = (tmp_path / "out" / "README.md").read_text()
    assert "split: train\n    path: train.jsonl" in card
    assert "split: test" not in card  # empty splits would break load_dataset
    assert len(read_labels_file(tmp_path / "out" / "labels.jsonl")) == 1


def test_push_uploads_the_folder_to_a_private_dataset_repo(db, profile, tmp_path, monkeypatch):
    import huggingface_hub

    calls = {}

    class FakeApi:
        def create_repo(self, repo_id, **kwargs):
            calls["create"] = (repo_id, kwargs)
            return f"https://huggingface.co/datasets/{repo_id}"

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    classify(db, "1", 0.9, "accept")
    export_dataset(db, profile, tmp_path / "out")

    url = push_dataset(tmp_path / "out", "me/events")

    assert url == "https://huggingface.co/datasets/me/events"
    assert calls["create"] == ("me/events", {"repo_type": "dataset", "private": True, "exist_ok": True})
    assert calls["upload"]["repo_type"] == "dataset"
    assert calls["upload"]["commit_message"] == "Export: 1 train, 0 test papers"


def test_push_errors_become_readable(db, profile, tmp_path, monkeypatch):
    import huggingface_hub
    import pytest

    class NoTokenApi:
        def create_repo(self, *args, **kwargs):
            raise OSError("no token")

    monkeypatch.setattr(huggingface_hub, "HfApi", NoTokenApi)
    export_dataset(db, profile, tmp_path / "out")

    with pytest.raises(RuntimeError, match="hf auth login"):
        push_dataset(tmp_path / "out", "me/events")


def test_cli_labels_export_and_import(tmp_path):
    config = tmp_path / "profile.yaml"
    config.write_text(
        f"topic: Test\ndata_dir: {tmp_path / 'data'}\nsource:\n  categories: [cs.CV]\n"
        "stages:\n  - name: relevance\n    type: noul\n    instructions: Relevant?\n"
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"paper_id": "1", "relevant": true, "split": "train", "labelled_at": "2026-10-01T00:00:00+00:00"}\n')

    assert main(["labels", "import", str(labels), "--config", str(config)]) == 0
    assert main(["labels", "export", "--config", str(config)]) == 0
    assert read_labels_file(tmp_path / "data" / "labels.jsonl")[0]["paper_id"] == "1"


def test_cli_labels_import_defaults_to_the_exported_file_and_explains_a_missing_one(tmp_path, caplog):
    config = tmp_path / "profile.yaml"
    config.write_text(
        f"topic: Test\ndata_dir: {tmp_path / 'data'}\nsource:\n  categories: [cs.CV]\n"
        "stages:\n  - name: relevance\n    type: noul\n    instructions: Relevant?\n"
    )

    assert main(["labels", "import", "labels.jsonl", "--config", str(config)]) == 1
    assert "No labels file at labels.jsonl" in caplog.text

    assert main(["labels", "export", "--config", str(config)]) == 0
    assert main(["labels", "import", "--config", str(config)]) == 0
