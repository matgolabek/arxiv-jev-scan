"""Score a decision backend (Jev or a local model) on a held-out dataset file,
asking the questions the way a scan does: the relevance question alone, and
the later stages one at a time for relevant papers only.

Asking one question at a time matters for models such as Clef that read all
questions of a request together. If relevant papers came with an extra
sub-topic question attached, the model could tell their relevance from the
number of questions instead of from the abstract.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from scanner.config import Profile
from scanner.exporters.dataset import relevance_stage
from scanner.jev_client import JevClient
from scanner.pipeline import decide


@dataclass
class Prediction:
    paper_id: str
    title: str
    relevant: bool          # gold answer
    label_source: str       # human / jev
    probability: float
    decision: str           # accept / maybe / reject
    stage_hits: dict[str, bool] = field(default_factory=dict)  # later stage -> matches gold


@dataclass
class Report:
    predictions: list[Prediction]
    input_tokens: int = 0
    cost_usd: float = 0.0

    def scores(self, label_source: str | None = None) -> dict:
        rows = [p for p in self.predictions if label_source in (None, p.label_source)]
        relevant = [p for p in rows if p.relevant]
        accepted = [p for p in rows if p.decision == "accept"]
        hits = [hit for p in relevant for hit in p.stage_hits.values()]
        return {
            "papers": len(rows),
            "relevant": len(relevant),
            "found": _share(sum(p.decision != "reject" for p in relevant), len(relevant)),
            "precision": _share(sum(p.relevant for p in accepted), len(accepted)),
            "maybe": _share(sum(p.decision == "maybe" for p in rows), len(rows)),
            "sub_topics": _share(sum(hits), len(hits)),
        }

    def missed(self) -> list[Prediction]:
        return [p for p in self.predictions if p.relevant and p.decision == "reject"]

    def false_accepts(self) -> list[Prediction]:
        return [p for p in self.predictions if not p.relevant and p.decision == "accept"]


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def evaluate(profile: Profile, client: JevClient, path: Path, workers: int = 4) -> Report:
    relevance = relevance_stage(profile)
    later = [stage for stage in profile.stages if stage is not relevance]
    with Path(path).open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise RuntimeError(f"{path} has no papers. Export with --test-size first.")
    report = Report(predictions=[])

    def score_row(row: dict) -> tuple[Prediction, int, float]:
        state = json.loads(row["state"])
        gold = json.loads(row["gold"])
        answer = client.ask(relevance, state)
        tokens, cost = answer.input_tokens, answer.cost_usd
        prediction = Prediction(
            paper_id=row["id"],
            title=state.get("title", ""),
            relevant=gold[relevance.name] is True,
            label_source=row.get("label_source", "jev"),
            probability=answer.result["probability"],
            decision=decide(relevance, answer.result),
        )
        # Sub-topics are only asked about relevant papers, as in a scan; scored where gold has them.
        for stage in later:
            if not prediction.relevant or stage.name not in gold:
                continue
            answer = client.ask(stage, state)
            tokens, cost = tokens + answer.input_tokens, cost + answer.cost_usd
            prediction.stage_hits[stage.name] = _matches(stage.type, answer.result, gold[stage.name])
        return prediction, tokens, cost

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for prediction, tokens, cost in pool.map(score_row, rows):
            report.predictions.append(prediction)
            report.input_tokens += tokens
            report.cost_usd += cost
    return report


def _matches(question_type: str, result: dict, gold) -> bool:
    if question_type == "noul":
        return (result["probability"] >= 0.5) == gold
    label = gold["label"] if isinstance(gold, dict) else gold
    return result["selected"] == label
