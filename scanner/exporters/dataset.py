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

test.jsonl holds the papers you labelled from the test sample, plus, with
`test_size`, a random sample held out with Jev's answers (half relevant, half
not). Held-out papers are remembered in the database, so later exports keep
them in the test set even after new papers are scanned. Everything else goes
to train.jsonl. Never train on the test file: it is what tells you how a model
compares with Jev.

The folder also gets labels.jsonl (a backup of your labels, see
`write_labels_file`) and a README.md dataset card, so it can be pushed to a
Hugging Face dataset repository as is and loaded with `load_dataset`.
"""

import json
import random
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
    test_sources: Counter = field(default_factory=Counter)   # human / jev, test rows only
    questions: Counter = field(default_factory=Counter)      # stage name -> rows that ask it
    skipped_maybe: int = 0                                   # unlabelled "maybe" papers


def relevance_stage(profile: Profile) -> Stage:
    """The first yes/no stage, the one that decides whether a paper is on topic."""
    for stage in profile.stages:
        if stage.type == "noul":
            return stage
    raise RuntimeError("The config has no yes/no (noul) stage, so there is no relevance question to train.")


def build_rows(
    db: Database, profile: Profile, test_size: int | None = None,
) -> tuple[list[dict], DatasetSummary]:
    """Dataset rows, one per paper with a known relevance answer.

    With `test_size`, tops the test set up to that many papers (half relevant)
    by holding out random training papers, and records them in the database.
    """
    relevance = relevance_stage(profile)
    later_stages = [stage for stage in profile.stages if stage is not relevance]
    labels = db.get_labels()
    holdout = db.get_holdout()
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

        if split == "train" and paper["paper_id"] in holdout:
            split = "test"
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
            "_relevant": relevant,
        })

    if test_size:
        db.add_to_holdout(hold_out(rows, test_size))

    for row in rows:
        split, relevant = row["split"], row.pop("_relevant")
        summary.rows[split] += 1
        summary.relevant[split] += relevant
        summary.label_sources[row["label_source"]] += 1
        if split == "test":
            summary.test_sources[row["label_source"]] += 1
        summary.questions.update(json.loads(row["questions"]).keys())
    return rows, summary


def hold_out(rows: list[dict], test_size: int) -> list[str]:
    """Move random training rows to the test split until it holds `test_size`
    papers, half of them relevant. Returns the ids that were moved.

    At most half of all relevant papers go to the test set, so training keeps
    enough of them.
    """
    targets = {True: test_size // 2, False: test_size - test_size // 2}
    targets[True] = min(targets[True], sum(row["_relevant"] for row in rows) // 2)
    rng = random.Random(0)
    moved = []
    for relevant, target in targets.items():
        have = sum(1 for row in rows if row["split"] == "test" and row["_relevant"] == relevant)
        candidates = [row for row in rows if row["split"] == "train" and row["_relevant"] == relevant]
        for row in rng.sample(candidates, max(0, min(target - have, len(candidates)))):
            row["split"] = "test"
            moved.append(row["id"])
    return moved


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


def export_dataset(
    db: Database, profile: Profile, out_dir: Path | None = None, test_size: int | None = None,
) -> DatasetSummary:
    """Write train.jsonl, test.jsonl, manifest.json, labels.jsonl and README.md.
    Returns what was written."""
    out_dir = Path(out_dir or profile.dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_rows(db, profile, test_size)

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
        "test_label_sources": dict(summary.test_sources),
        "questions": dict(summary.questions),
        "skipped_unlabelled_maybe": summary.skipped_maybe,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(dataset_card(profile, summary), encoding="utf-8")
    write_labels_file(db, out_dir / "labels.jsonl")
    return summary


def dataset_card(profile: Profile, summary: DatasetSummary) -> str:
    """README.md for the Hugging Face Hub. The `configs` header tells
    `load_dataset` which files are the splits, so labels.jsonl and
    manifest.json are not mistaken for data. Empty splits are left out
    because the Hub cannot load an empty file."""
    splits = [split for split in ("train", "test") if summary.rows[split]]
    data_files = "".join(f"  - split: {split}\n    path: {split}.jsonl\n" for split in splits)
    header = f"---\nconfigs:\n- config_name: default\n  data_files:\n{data_files}---\n\n" if splits else ""
    counts = "\n".join(
        f"| {split} | {summary.rows[split]} | {summary.relevant[split]} |" for split in ("train", "test")
    )
    return f"""{header}# {profile.topic}: arXiv relevance dataset

Made with [arxiv-jev-scan](https://github.com/matgolabek/arxiv-jev-scan) (`export-dataset`).
Rows follow the format of Unsloth's decision-model trainer: `state` (title and abstract),
`questions` and `gold`, each a JSON string. Gold relevance is a human label where there is one,
otherwise Jev's decision (see `label_source`). `test` is held out and must not be trained on.

| split | papers | relevant |
|---|---|---|
{counts}

Also here: `labels.jsonl` (the human labels, for `python -m scanner.cli labels import`) and
`manifest.json` (counts). Titles and abstracts are arXiv metadata, released under CC0.
"""


def write_labels_file(db: Database, path: Path) -> int:
    """Save your labels as JSON lines. Returns how many were written."""
    labels = db.export_labels()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        for label in labels:
            file.write(json.dumps(label, ensure_ascii=False) + "\n")
    return len(labels)


def read_labels_file(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def push_dataset(out_dir: Path, repo_id: str, public: bool = False) -> str:
    """Upload the exported folder to a Hugging Face dataset repository,
    creating it (private unless `public`) if needed. Returns its URL.

    Log in once with `hf auth login`, or set HF_TOKEN.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("Pushing needs huggingface_hub: pip install huggingface_hub") from None

    manifest = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    api = HfApi()
    try:
        url = api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(out_dir),
            allow_patterns=["*.jsonl", "manifest.json", "README.md"],
            commit_message=(
                f"Export: {files['train.jsonl']['rows']} train, {files['test.jsonl']['rows']} test papers"
            ),
        )
    except Exception as error:  # not logged in, no access, network...
        raise RuntimeError(
            f"Could not push to {repo_id}: {error}. Log in with 'hf auth login' or set HF_TOKEN."
        ) from error
    return str(url)
