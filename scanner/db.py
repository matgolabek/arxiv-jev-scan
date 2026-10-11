"""SQLite storage, the source of truth: every paper we have seen, every Jev
answer, every scan run (so an interrupted scan can resume) and your own
relevance labels (for building a training dataset)."""

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from scanner.models import Paper, StageResult

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id          TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    title             TEXT NOT NULL,
    authors           TEXT NOT NULL,     -- JSON list
    abstract          TEXT NOT NULL,
    published_date    TEXT,              -- first version, YYYY-MM-DD
    categories        TEXT,              -- JSON list
    url               TEXT,
    pdf_url           TEXT,
    first_seen_at     TEXT NOT NULL,
    final_status      TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | maybe | rejected
    relevance         REAL,              -- probability from the first noul stage
    tags              TEXT,              -- JSON {tag: probability} from choice stages
    pdf_path          TEXT,
    pdf_downloaded_at TEXT
);

CREATE TABLE IF NOT EXISTS stage_results (
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id),
    stage_name    TEXT NOT NULL,
    question_type TEXT NOT NULL,         -- noul | choice
    result        TEXT NOT NULL,         -- JSON answer
    decision      TEXT NOT NULL,         -- accept | maybe | reject
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    model         TEXT,
    ran_at        TEXT NOT NULL,
    PRIMARY KEY (paper_id, stage_name)
);

CREATE TABLE IF NOT EXISTS scan_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    source       TEXT NOT NULL,
    query_params TEXT NOT NULL,          -- JSON: categories, since, until
    checkpoint   TEXT,                   -- last fully processed day
    status       TEXT NOT NULL           -- running | completed | interrupted
);

-- Your answer to the relevance question, which overrides Jev's in the dataset.
-- split 'test' marks the hand-checked papers held out for evaluation.
CREATE TABLE IF NOT EXISTS labels (
    paper_id    TEXT PRIMARY KEY REFERENCES papers(paper_id),
    relevant    INTEGER NOT NULL,        -- 1 yes, 0 no
    split       TEXT NOT NULL,           -- train | test
    labelled_at TEXT NOT NULL
);

-- Papers held out for the test set with Jev's answer as gold (export-dataset
-- --test-size). Stored so a paper, once held out, never moves into training.
CREATE TABLE IF NOT EXISTS test_holdout (
    paper_id  TEXT PRIMARY KEY REFERENCES papers(paper_id),
    chosen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS papers_by_status ON papers(final_status);
"""

STATUSES = ("pending", "accepted", "maybe", "rejected")
CLASSIFIED = ("accepted", "maybe", "rejected")
SPLITS = ("train", "test")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- papers ---------------------------------------------------------

    def add_new_papers(self, papers: list[Paper]) -> list[Paper]:
        """Store papers we have never seen and return only those.

        This is the dedup gate: a paper that is already in the database is
        ignored here, so it is never sent to Jev a second time.
        """
        new = []
        with self.conn:
            for paper in papers:
                cursor = self.conn.execute(
                    """INSERT OR IGNORE INTO papers
                       (paper_id, source, title, authors, abstract, published_date,
                        categories, url, pdf_url, first_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        paper.paper_id, paper.source, paper.title, json.dumps(paper.authors),
                        paper.abstract, paper.published, json.dumps(paper.categories),
                        paper.url, paper.pdf_url, now(),
                    ),
                )
                if cursor.rowcount == 1:
                    new.append(paper)
        return new

    def pending_papers(self) -> list[Paper]:
        """Papers stored but not classified yet (e.g. after a crash or API error)."""
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE final_status = 'pending' ORDER BY paper_id"
        )
        return [
            Paper(
                paper_id=row["paper_id"],
                source=row["source"],
                title=row["title"],
                authors=json.loads(row["authors"]),
                abstract=row["abstract"],
                published=row["published_date"],
                categories=json.loads(row["categories"]),
                url=row["url"],
                pdf_url=row["pdf_url"],
            )
            for row in rows
        ]

    def save_classification(
        self,
        paper_id: str,
        results: list[StageResult],
        status: str,
        relevance: float | None,
        tags: dict[str, float],
    ) -> None:
        """Store every stage result of one paper and its final status together."""
        with self.conn:
            for result in results:
                self.conn.execute(
                    """INSERT OR REPLACE INTO stage_results
                       (paper_id, stage_name, question_type, result, decision,
                        input_tokens, output_tokens, cost_usd, model, ran_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        paper_id, result.stage_name, result.question_type,
                        json.dumps(result.result), result.decision, result.input_tokens,
                        result.output_tokens, result.cost_usd, result.model, now(),
                    ),
                )
            self.conn.execute(
                "UPDATE papers SET final_status = ?, relevance = ?, tags = ? WHERE paper_id = ?",
                (status, relevance, json.dumps(tags), paper_id),
            )

    def get_papers(self, statuses: list[str] | None = None) -> list[dict]:
        """Papers as dicts (JSON columns decoded), newest first."""
        query = "SELECT * FROM papers"
        params: list[str] = []
        if statuses:
            placeholders = ", ".join(["?"] * len(statuses))
            query += f" WHERE final_status IN ({placeholders})"
            params = list(statuses)
        query += " ORDER BY published_date DESC, paper_id DESC"
        return [_decode_paper(row) for row in self.conn.execute(query, params)]

    def get_stage_results(self, paper_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM stage_results WHERE paper_id = ? ORDER BY rowid", (paper_id,)
        )
        return [dict(row) | {"result": json.loads(row["result"])} for row in rows]

    def get_all_stage_results(self) -> dict[str, dict[str, dict]]:
        """paper_id -> {stage name: stage result}, for every classified paper."""
        results: dict[str, dict[str, dict]] = {}
        for row in self.conn.execute("SELECT * FROM stage_results ORDER BY rowid"):
            result = dict(row) | {"result": json.loads(row["result"])}
            results.setdefault(row["paper_id"], {})[row["stage_name"]] = result
        return results

    def mark_downloaded(self, paper_id: str, pdf_path: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE papers SET pdf_path = ?, pdf_downloaded_at = ? WHERE paper_id = ?",
                (pdf_path, now(), paper_id),
            )

    def papers_missing_pdf(self, statuses: list[str]) -> list[dict]:
        return [paper for paper in self.get_papers(statuses) if not paper["pdf_path"]]

    def totals(self) -> dict:
        """Paper counts per status plus total token usage and cost."""
        totals = {status: 0 for status in STATUSES}
        for row in self.conn.execute(
            "SELECT final_status, COUNT(*) AS n FROM papers GROUP BY final_status"
        ):
            totals[row["final_status"]] = row["n"]
        totals["papers"] = sum(totals.values())

        usage = self.conn.execute(
            """SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                      COALESCE(SUM(cost_usd), 0), COUNT(*)
               FROM stage_results"""
        ).fetchone()
        totals["input_tokens"], totals["output_tokens"], totals["cost_usd"], totals["jev_calls"] = usage
        return totals

    # --- labels -------------------------------------------------------------

    def set_label(self, paper_id: str, relevant: bool, split: str | None = None) -> None:
        """Store your yes/no answer for a paper. Without `split`, a paper that is
        already labelled keeps its split and a new one goes to 'train'."""
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        with self.conn:
            self.conn.execute(
                """INSERT INTO labels (paper_id, relevant, split, labelled_at)
                   VALUES (?, ?, COALESCE(?, 'train'), ?)
                   ON CONFLICT (paper_id) DO UPDATE SET
                       relevant = excluded.relevant,
                       split = COALESCE(?, labels.split),
                       labelled_at = excluded.labelled_at""",
                (paper_id, int(relevant), split, now(), split),
            )

    def import_labels(self, labels: list[dict]) -> dict[str, int]:
        """Merge labels saved by `export_labels`, e.g. on a new machine or after
        losing the database. Where both sides labelled a paper, the newer label
        wins. Labels for papers not scanned yet are kept and apply once they are.
        """
        existing = self.get_labels()
        counts = {"added": 0, "updated": 0, "unchanged": 0}
        with self.conn:
            for label in labels:
                if label["split"] not in SPLITS:
                    raise ValueError(f"{label['paper_id']}: split must be one of {SPLITS}")
                current = existing.get(label["paper_id"])
                if current and current["labelled_at"] >= label["labelled_at"]:
                    counts["unchanged"] += 1
                    continue
                self.conn.execute(
                    """INSERT OR REPLACE INTO labels (paper_id, relevant, split, labelled_at)
                       VALUES (?, ?, ?, ?)""",
                    (label["paper_id"], int(label["relevant"]), label["split"], label["labelled_at"]),
                )
                counts["updated" if current else "added"] += 1
        return counts

    def export_labels(self) -> list[dict]:
        """All labels with the paper title (for readability), oldest first."""
        rows = self.conn.execute(
            """SELECT l.*, p.title FROM labels l LEFT JOIN papers p USING (paper_id)
               ORDER BY l.labelled_at, l.paper_id"""
        )
        return [
            {
                "paper_id": row["paper_id"],
                "relevant": bool(row["relevant"]),
                "split": row["split"],
                "labelled_at": row["labelled_at"],
                "title": row["title"],
            }
            for row in rows
        ]

    def get_holdout(self) -> set[str]:
        return {row[0] for row in self.conn.execute("SELECT paper_id FROM test_holdout")}

    def add_to_holdout(self, paper_ids: list[str]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO test_holdout (paper_id, chosen_at) VALUES (?, ?)",
                [(paper_id, now()) for paper_id in paper_ids],
            )

    def remove_label(self, paper_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM labels WHERE paper_id = ?", (paper_id,))

    def get_labels(self) -> dict[str, dict]:
        """paper_id -> {relevant: bool, split, labelled_at}"""
        return {
            row["paper_id"]: {
                "relevant": bool(row["relevant"]),
                "split": row["split"],
                "labelled_at": row["labelled_at"],
            }
            for row in self.conn.execute("SELECT * FROM labels")
        }

    def label_counts(self) -> dict:
        """How many papers you labelled, per split and per Jev status."""
        counts = {
            "train": 0, "test": 0, "maybe_left": 0,
            "test_by_status": {status: 0 for status in CLASSIFIED},
        }
        for row in self.conn.execute(
            """SELECT l.split, p.final_status, COUNT(*) AS n
               FROM labels l JOIN papers p USING (paper_id) GROUP BY l.split, p.final_status"""
        ):
            counts[row["split"]] += row["n"]
            if row["split"] == "test" and row["final_status"] in CLASSIFIED:
                counts["test_by_status"][row["final_status"]] += row["n"]
        counts["maybe_left"] = self.conn.execute(
            """SELECT COUNT(*) FROM papers
               WHERE final_status = 'maybe'
                 AND paper_id NOT IN (SELECT paper_id FROM labels)"""
        ).fetchone()[0]
        return counts

    def next_unlabelled_maybe(self, skip: list[str] = ()) -> dict | None:
        """The most likely relevant 'maybe' paper you have not labelled yet."""
        placeholders = ", ".join(["?"] * len(skip))
        row = self.conn.execute(
            f"""SELECT * FROM papers
                WHERE final_status = 'maybe'
                  AND paper_id NOT IN (SELECT paper_id FROM labels)
                  AND paper_id NOT IN ({placeholders})
                ORDER BY relevance DESC, paper_id LIMIT 1""",
            list(skip),
        ).fetchone()
        return _decode_paper(row) if row else None

    def next_test_candidate(self, skip: list[str] = ()) -> dict | None:
        """A random unlabelled paper for the test set, drawn from the Jev status
        (accepted, maybe, rejected) that has the fewest test labels so far.

        Mostly rejected papers come out of a scan, so a plain random sample
        would hold almost no relevant ones. Balancing by status keeps enough of
        each kind to measure how many relevant papers a model misses.
        """
        counts = self.label_counts()["test_by_status"]
        placeholders = ", ".join(["?"] * len(skip))
        for status in sorted(CLASSIFIED, key=lambda status: counts[status]):
            row = self.conn.execute(
                f"""SELECT * FROM papers
                    WHERE final_status = ?
                      AND paper_id NOT IN (SELECT paper_id FROM labels)
                      AND paper_id NOT IN ({placeholders})
                    ORDER BY random() LIMIT 1""",
                [status, *skip],
            ).fetchone()
            if row:
                return _decode_paper(row)
        return None

    # --- scan runs --------------------------------------------------------

    def start_run(self, source: str, categories: list[str], since: date, until: date) -> int:
        params = {
            "categories": sorted(categories),
            "since": since.isoformat(),
            "until": until.isoformat(),
        }
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO scan_runs (started_at, source, query_params, status)
                   VALUES (?, ?, ?, 'running')""",
                (now(), source, json.dumps(params)),
            )
        return cursor.lastrowid

    def find_unfinished_run(self, source: str, categories: list[str], since: date) -> dict | None:
        """The newest scan with the same source, categories and start date that
        did not complete, so it can be resumed from its checkpoint."""
        rows = self.conn.execute(
            """SELECT * FROM scan_runs
               WHERE source = ? AND status != 'completed' ORDER BY run_id DESC""",
            (source,),
        )
        for row in rows:
            params = json.loads(row["query_params"])
            if params["categories"] == sorted(categories) and params["since"] == since.isoformat():
                return dict(row)
        return None

    def set_run_status(self, run_id: int, status: str) -> None:
        finished_at = None if status == "running" else now()
        with self.conn:
            self.conn.execute(
                "UPDATE scan_runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, finished_at, run_id),
            )

    def save_checkpoint(self, run_id: int, day: date) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE scan_runs SET checkpoint = ? WHERE run_id = ?", (day.isoformat(), run_id)
            )

    def last_scanned_day(self, source: str, categories: list[str]) -> date | None:
        """Latest day fully covered by a completed scan of these categories."""
        rows = self.conn.execute(
            """SELECT query_params, checkpoint FROM scan_runs
               WHERE source = ? AND status = 'completed' AND checkpoint IS NOT NULL""",
            (source,),
        )
        days = [
            date.fromisoformat(row["checkpoint"])
            for row in rows
            if json.loads(row["query_params"])["categories"] == sorted(categories)
        ]
        return max(days, default=None)

    def get_runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM scan_runs ORDER BY run_id DESC LIMIT ?", (limit,))
        return [dict(row) | {"query_params": json.loads(row["query_params"])} for row in rows]


def _decode_paper(row: sqlite3.Row) -> dict:
    paper = dict(row)
    paper["authors"] = json.loads(paper["authors"])
    paper["categories"] = json.loads(paper["categories"] or "[]")
    paper["tags"] = json.loads(paper["tags"] or "{}")
    return paper
