"""Export classified papers as a training dataset for a local decision model.

The rows follow the format of Unsloth's decision-model trainer (the same as
Laya and Clef datasets): `state`, `questions` and `gold`, each stored as a
JSON string. https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth

Where the gold answers come from:

- The relevance question (the first yes/no stage): your own label if you gave
  one, otherwise Jev's decision. Papers Jev left at "maybe" that you have not
  labelled are skipped, since nobody knows their answer.
- Later stages: Jev's answers, only for papers whose gold relevance is yes,
  because their questions assume an on-topic paper. Choice stages keep Jev's
  full probabilities as soft labels; yes/no stages that ended at "maybe" are
  left out.

Papers you labelled from the test sample go to test.jsonl, everything else to
train.jsonl. Never train on the test file: it is what tells you how a model
compares with Jev.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from scanner.config import Profile, Stage
from scanner.db import CLASSIFIED, Database, now
from scanner.models import paper_state


@dataclass
class DatasetSummary:
    rows: dict[str, int] = field(default_factory=lambda: {"train": 0, "test": 0})
    relevant: dict[str, int] = field(default_factory=lambda: {"train": 0, "test": 0})
    label_sources: Counter = field(default_factory=Counter)  # human / jev
    questions: Counter = field(default_factory=Counter)      # stage name -> rows that ask it
    skipped_maybe: int = 0                                   # unlabelled "maybe" papers


def relevance_stage(profile: Profile) -> Stage:
    """The first yes/no stage, the one that decides whether a paper is on topic."""
    for stage in profile.stages:
        if stage.type == "noul":
            return stage
    raise RuntimeError("The config has no yes/no (noul) stage, so there is no relevance question to train.")


def build_rows(db: Database, profile: Profile) -> tuple[list[dict], DatasetSummary]:
    relevance = relevance_stage(profile)
    later_stages = [stage for stage in profile.stages if stage is not relevance]
    labels = db.get_labels()
    all_results = db.get_all_stage_results()
    summary = DatasetSummary()
    rows = []

    for paper in sorted(db.get_papers(list(CLASSIFIED)), key=lambda paper: paper["paper_id"]):
        results = all_results.get(paper["paper_id"], {})
        label = labels.get(paper["paper_id"])
        if label:
            relevant, source, split = label["relevant"], "human", label["split"]
        else:
            decision = results.get(relevance.name, {}).get("decision")
            if decision not in ("accept", "reject"):
                summary.skipped_maybe += decision == "maybe"
                continue
            relevant, source, split = decision == "accept", "jev", "train"

        questions = {relevance.name: relevance.question()}
        gold = {relevance.name: relevant}
        if relevant:
            for stage in later_stages:
                answer = stage_gold(stage, results.get(stage.name))
                if answer is not None:
                    questions[stage.name] = stage.question()
                    gold[stage.name] = answer

        rows.append({
            "id": paper["paper_id"],
            "split": split,
            "label_source": source,
            "published": paper["published_date"],
            "state": json.dumps(paper_state(paper["title"], paper["abstract"]), ensure_ascii=False),
            "questions": json.dumps(questions, ensure_ascii=False),
            "gold": json.dumps(gold, ensure_ascii=False),
            # Jev's raw answers, for comparing models later. Not used as gold.
            "jev": json.dumps({name: result["result"] for name, result in results.items()}),
        })
        summary.rows[split] += 1
        summary.relevant[split] += relevant
        summary.label_sources[source] += 1
        summary.questions.update(questions.keys())
    return rows, summary


def stage_gold(stage: Stage, result: dict | None) -> bool | dict | None:
    """Gold answer of a later stage taken from Jev, or None to leave it out."""
    if result is None:
        return None
    if stage.type == "noul":
        if result["decision"] == "maybe":
            return None
        return result["decision"] == "accept"
    answer = result["result"]
    return {"type": "choice", "label": answer["selected"], "probabilities": answer["probabilities"]}


def export_dataset(db: Database, profile: Profile, out_dir: Path | None = None) -> DatasetSummary:
    """Write train.jsonl, test.jsonl and manifest.json. Returns what was written."""
    out_dir = Path(out_dir or profile.dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_rows(db, profile)

    for split in ("train", "test"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                if row["split"] == split:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "topic": profile.topic,
        "created_at": now(),
        "format": "Unsloth decision dataset: state, questions and gold as JSON strings",
        "relevance_question": relevance_stage(profile).name,
        "files": {
            f"{split}.jsonl": {
                "rows": summary.rows[split],
                "relevant": summary.relevant[split],
                "not_relevant": summary.rows[split] - summary.relevant[split],
            }
            for split in ("train", "test")
        },
        "label_sources": dict(summary.label_sources),
        "questions": dict(summary.questions),
        "skipped_unlabelled_maybe": summary.skipped_maybe,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary
