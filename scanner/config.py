"""Load a scan profile from YAML: the topic, where its data lives, which
arXiv categories to scan and the ordered list of Jev stages."""

from dataclasses import dataclass
from pathlib import Path

import yaml

QUESTION_TYPES = ("noul", "choice")


@dataclass
class Stage:
    name: str
    type: str                       # "noul" (yes/no) or "choice" (pick one option)
    instructions: str
    criteria: dict | None = None
    # noul: >= accept_threshold is accept, <= reject_threshold is reject, in between is maybe
    accept_threshold: float = 0.8
    reject_threshold: float = 0.2
    # choice: options with at least this probability become tags of the paper
    tag_threshold: float = 0.25
    # choice: answers below this confidence are flagged as maybe (they never reject)
    min_confidence: float = 0.5

    def question(self) -> dict:
        """The question in the format Jev (and decision-model datasets) expect."""
        question = {"type": self.type, "instructions": self.instructions}
        if self.criteria:
            question["criteria"] = self.criteria
        return question


@dataclass
class Profile:
    topic: str
    data_dir: Path
    categories: list[str]
    stages: list[Stage]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "papers.db"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def obsidian_dir(self) -> Path:
        return self.data_dir / "obsidian"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"


def load_profile(path: str | Path) -> Profile:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stages = [_load_stage(item) for item in raw["stages"]]

    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: stage names must be unique, got {names}")

    return Profile(
        topic=str(raw["topic"]).strip(),
        data_dir=Path(raw.get("data_dir", "data")),
        categories=list(raw["source"]["categories"]),
        stages=stages,
    )


def _load_stage(item: dict) -> Stage:
    try:
        stage = Stage(**item)
    except TypeError as error:
        raise ValueError(f"stage {item.get('name', '?')}: {error}") from None

    if stage.type not in QUESTION_TYPES:
        raise ValueError(f"stage {stage.name}: type must be one of {QUESTION_TYPES}")
    if stage.type == "choice" and not stage.criteria:
        raise ValueError(f"stage {stage.name}: a choice stage needs criteria (its options)")
    if stage.type == "noul" and not stage.reject_threshold < stage.accept_threshold:
        raise ValueError(f"stage {stage.name}: reject_threshold must be below accept_threshold")

    if stage.criteria:
        stage.criteria = _string_keys(stage.criteria)
    return stage


def _string_keys(criteria: dict) -> dict:
    # YAML reads the keys `true:` and `false:` as booleans, but Jev expects strings.
    fixed = {}
    for key, value in criteria.items():
        if isinstance(key, bool):
            key = "true" if key else "false"
        fixed[str(key)] = value
    return fixed
