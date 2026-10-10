"""The scan pipeline: fetch papers day by day, skip the ones already in the
database and run each new paper through the configured Jev stages."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from scanner.config import Profile, Stage
from scanner.db import Database
from scanner.jev_client import FATAL_ERRORS, JevClient
from scanner.models import Paper, Progress, StageResult, paper_state
from scanner.sources.base import PaperSource

log = logging.getLogger(__name__)

# A harvest also returns new versions of older papers. We keep papers whose
# first version is at most this many days older than the scan start, so papers
# that waited in arXiv's moderation queue are not lost, and skip the rest.
SLACK_DAYS = 30


@dataclass
class ScanSummary:
    fetched: int = 0
    new: int = 0
    accepted: int = 0
    maybe: int = 0
    rejected: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    accepted_titles: list[str] = field(default_factory=list)


def decide(stage: Stage, result: dict) -> str:
    """Turn a Jev answer into accept / maybe / reject."""
    if stage.type == "noul":
        probability = result["probability"]
        if probability >= stage.accept_threshold:
            return "accept"
        if probability <= stage.reject_threshold:
            return "reject"
        return "maybe"
    # Choice stages label papers and never reject; low confidence is only flagged.
    return "accept" if result["confidence"] >= stage.min_confidence else "maybe"


def paper_status(results: list[StageResult]) -> str:
    """Final status of a paper, decided by its yes/no (noul) stages."""
    decisions = [r.decision for r in results if r.question_type == "noul"]
    if "reject" in decisions:
        return "rejected"
    if "maybe" in decisions:
        return "maybe"
    return "accepted"


def relevance_of(results: list[StageResult]) -> float | None:
    """Probability from the first noul stage, used for sorting and filtering."""
    for result in results:
        if result.question_type == "noul":
            return result.result["probability"]
    return None


def tags_of(stages: list[Stage], results: list[StageResult]) -> dict[str, float]:
    """Tags from all choice stages: the selected option plus every option at
    least `tag_threshold` likely, sorted from most to least likely."""
    thresholds = {stage.name: stage.tag_threshold for stage in stages}
    tags: dict[str, float] = {}
    for result in results:
        if result.question_type != "choice":
            continue
        threshold = thresholds.get(result.stage_name, 0.25)
        for option, probability in result.result["probabilities"].items():
            if probability >= threshold or option == result.result["selected"]:
                tags[option] = max(probability, tags.get(option, 0.0))
    return dict(sorted(tags.items(), key=lambda item: item[1], reverse=True))


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def since_last_run(db: Database, source: str, categories: list[str]) -> date:
    """Start date for an incremental scan: the last day a completed scan covered.

    That day is scanned again because it may have been only partly published
    at the time; the dedup gate makes this free.
    """
    last_day = db.last_scanned_day(source, categories)
    if last_day:
        return last_day
    fallback = utc_today() - timedelta(days=3)
    log.info("No earlier scan of %s, starting from %s", ", ".join(categories), fallback)
    return fallback


class Pipeline:
    def __init__(
        self,
        profile: Profile,
        db: Database,
        jev: JevClient,
        source: PaperSource,
        workers: int = 10,
    ):
        self.profile = profile
        self.db = db
        self.jev = jev
        self.source = source
        self.workers = workers

    def classify(self, paper: Paper) -> list[StageResult]:
        """Run the stages in order for one paper.

        Stops at the first reject, so most papers cost a single Jev call.
        """
        state = paper_state(paper.title, paper.abstract)
        results = []
        for stage in self.profile.stages:
            answer = self.jev.ask(stage, state)
            decision = decide(stage, answer.result)
            results.append(
                StageResult(
                    stage_name=stage.name,
                    question_type=stage.type,
                    result=answer.result,
                    decision=decision,
                    input_tokens=answer.input_tokens,
                    output_tokens=answer.output_tokens,
                    cost_usd=answer.cost_usd,
                    model=answer.model,
                )
            )
            if decision == "reject":
                break
        return results

    def scan(
        self,
        since: date,
        until: date | None = None,
        categories: list[str] | None = None,
        max_papers: int | None = None,
        progress: Progress | None = None,
    ) -> ScanSummary:
        """Fetch and classify papers from `since` to `until` (inclusive).

        If an earlier scan with the same categories and start date was
        interrupted, it continues after the last finished day.
        """
        categories = categories or self.profile.categories
        until = min(until or utc_today(), utc_today())
        summary = ScanSummary()

        run = self.db.find_unfinished_run(self.source.name, categories, since)
        if run:
            run_id = run["run_id"]
            start = since
            if run["checkpoint"]:
                start = date.fromisoformat(run["checkpoint"]) + timedelta(days=1)
            self.db.set_run_status(run_id, "running")
            log.info("Resuming scan %d from %s", run_id, start)
        else:
            run_id = self.db.start_run(self.source.name, categories, since, until)
            start = since

        cutoff = (since - timedelta(days=SLACK_DAYS)).isoformat()
        days = [start + timedelta(days=n) for n in range((until - start).days + 1)]

        try:
            # Papers left pending by a crash or an API error go first.
            self._classify(self.db.pending_papers(), summary)

            for index, day in enumerate(days):
                # The part of the progress bar that belongs to this day.
                share = (index / len(days), (index + 1) / len(days))
                if progress:
                    progress(share[0], f"{day}: fetching papers")

                papers = [p for p in self.source.fetch_day(categories, day) if p.published >= cutoff]
                summary.fetched += len(papers)
                if max_papers is not None:
                    papers = papers[: max_papers - summary.new]

                new_papers = self.db.add_new_papers(papers)  # the dedup gate
                summary.new += len(new_papers)
                self._classify(new_papers, summary, progress, share, label=str(day))

                if max_papers is not None and summary.new >= max_papers:
                    log.info("Reached the limit of %d new papers, stopping", max_papers)
                    break
                self.db.save_checkpoint(run_id, day)
        except BaseException:  # also Ctrl+C, so the run is marked as resumable
            self.db.set_run_status(run_id, "interrupted")
            raise

        self.db.set_run_status(run_id, "completed")
        if progress:
            progress(1.0, "done")
        return summary

    def _classify(
        self,
        papers: list[Paper],
        summary: ScanSummary,
        progress: Progress | None = None,
        share: tuple[float, float] = (0.0, 1.0),
        label: str = "",
    ) -> None:
        """Classify papers in parallel. Jev calls run in worker threads, all
        database writes happen here in the calling thread.

        Progress is reported within `share`, the slice of the whole scan's
        progress bar that these papers represent.
        """
        if not papers:
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.classify, paper): paper for paper in papers}
            for done, future in enumerate(as_completed(futures), start=1):
                paper = futures[future]
                try:
                    results = future.result()
                except FATAL_ERRORS:
                    pool.shutdown(cancel_futures=True)
                    raise
                except Exception as error:
                    summary.failed += 1
                    log.warning("Could not classify %s, will retry next run: %s", paper.paper_id, error)
                    continue

                self._save(paper, results, summary)
                if progress:
                    start, end = share
                    fraction = start + (end - start) * done / len(papers)
                    progress(fraction, f"{label}: classified {done} of {len(papers)} new papers")

    def _save(self, paper: Paper, results: list[StageResult], summary: ScanSummary) -> None:
        status = paper_status(results)
        self.db.save_classification(
            paper.paper_id,
            results,
            status,
            relevance_of(results),
            tags_of(self.profile.stages, results),
        )
        summary.cost_usd += sum(r.cost_usd for r in results)
        if status == "accepted":
            summary.accepted += 1
            summary.accepted_titles.append(paper.title)
        elif status == "maybe":
            summary.maybe += 1
        else:
            summary.rejected += 1
