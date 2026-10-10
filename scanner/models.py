"""Plain data classes shared by the sources, the pipeline and the database."""

from collections.abc import Callable
from dataclasses import dataclass

# Progress callback used by long tasks: progress(fraction from 0 to 1, message)
Progress = Callable[[float, str], None]


def paper_state(title: str, abstract: str) -> dict:
    """What the model is shown about a paper. Training data must use the same
    state as the scan, so both build it here."""
    return {"title": title, "abstract": abstract}


@dataclass
class Paper:
    paper_id: str          # e.g. "2409.01234" for arXiv
    source: str            # name of the PaperSource that found it, e.g. "arxiv"
    title: str
    authors: list[str]
    abstract: str
    published: str         # date of the first version, YYYY-MM-DD
    categories: list[str]
    url: str               # landing page
    pdf_url: str


@dataclass
class StageResult:
    """Outcome of one pipeline stage (one Jev call) for one paper."""

    stage_name: str
    question_type: str     # "noul" or "choice"
    result: dict           # noul: {probability}, choice: {selected, probabilities, confidence}
    decision: str          # "accept", "maybe" or "reject"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
