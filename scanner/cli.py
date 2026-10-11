"""Command line interface. Run `python -m scanner.cli --help`."""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from scanner.config import load_profile
from scanner.db import Database
from scanner.exporters.dataset import export_dataset, push_dataset, read_labels_file, write_labels_file
from scanner.exporters.obsidian import export_accepted
from scanner.evaluate import evaluate
from scanner.exporters.dataset import relevance_stage
from scanner.jev_client import BACKENDS, DEFAULT_WORKERS, JevClient, backend_name
from scanner.notify import notify
from scanner.pdfs import download_all
from scanner.pipeline import Pipeline, since_last_run
from scanner.sources import ArxivSource

log = logging.getLogger("scanner")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        # One line per HTTP request would bury the scan's own progress.
        for name in ("httpx2", "typesafe_sdk"):
            logging.getLogger(name).setLevel(logging.WARNING)

    profile = load_profile(args.config)
    db = Database(profile.db_path)
    try:
        return args.command(args, profile, db)
    except KeyboardInterrupt:
        log.info("Stopped. Run the same command again to continue where it left off.")
        return 130
    except RuntimeError as error:
        log.error("%s", error)
        return 1
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    # Options shared by all commands, so they can go after the command name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config/stages.yaml", help="scan profile (default: %(default)s)")
    common.add_argument("-v", "--verbose", action="store_true", help="show debug logs")

    parser = argparse.ArgumentParser(
        prog="python -m scanner.cli",
        description="Find arXiv papers on your topic with TypeSafe's Jev model.",
    )
    commands = parser.add_subparsers(title="commands", dest="command_name", required=True)

    scan = commands.add_parser("scan", parents=[common], help="fetch new papers and classify them")
    scan.add_argument(
        "--category", action="append",
        help="arXiv category, e.g. cs.CV (repeat for more); default: from the config",
    )
    scan.add_argument(
        "--since", default="last-run",
        help="YYYY-MM-DD, or 'last-run' to continue after the last scan (default)",
    )
    scan.add_argument("--until", help="YYYY-MM-DD (default: today)")
    scan.add_argument("--max", type=int, help="stop after this many new papers, handy for a first try")
    scan.add_argument(
        "--workers", type=int, help="parallel requests (default: 10 for Jev, 2 for the local model)",
    )
    scan.add_argument("--no-notify", action="store_true", help="skip the desktop notification")
    scan.add_argument(
        "--backend", choices=BACKENDS,
        help="classify with Jev or the local model (default: DECISION_BACKEND from .env, else jev)",
    )
    scan.set_defaults(command=run_scan)

    download = commands.add_parser(
        "download-all", parents=[common], help="download PDFs of accepted papers"
    )
    download.add_argument(
        "--status", default="accepted",
        help="which papers: 'accepted' (default) or 'accepted,maybe'",
    )
    download.set_defaults(command=run_download_all)

    export = commands.add_parser("export", parents=[common], help="write Obsidian notes for accepted papers")
    export.set_defaults(command=run_export)

    dataset = commands.add_parser(
        "export-dataset", parents=[common],
        help="write a training dataset (train.jsonl, test.jsonl) for a local decision model",
    )
    dataset.add_argument("--out", help="output folder (default: <data_dir>/dataset)")
    dataset.add_argument(
        "--test-size", type=int, metavar="N",
        help="hold out random papers (half relevant) with Jev's answers until the test set has N",
    )
    dataset.add_argument(
        "--push", nargs="?", const="", metavar="USER/REPO",
        help="also upload it to a Hugging Face dataset repository: this one, or HF_DATASET_REPO "
             "from .env (log in first: hf auth login)",
    )
    dataset.add_argument(
        "--public", action="store_true", help="create the repository as public (default: private)",
    )
    dataset.set_defaults(command=run_export_dataset)

    labels = commands.add_parser("labels", help="save your labels to a file, or load them back")
    label_commands = labels.add_subparsers(title="label commands", dest="labels_command", required=True)
    labels_export = label_commands.add_parser(
        "export", parents=[common], help="write your labels to a JSON lines file",
    )
    labels_export.add_argument("--out", help="file to write (default: <data_dir>/labels.jsonl)")
    labels_export.set_defaults(command=run_labels_export)
    labels_import = label_commands.add_parser(
        "import", parents=[common], help="load labels from a file written by 'labels export'",
    )
    labels_import.add_argument(
        "file", nargs="?", help="labels file to load (default: <data_dir>/labels.jsonl)",
    )
    labels_import.set_defaults(command=run_labels_import)

    check = commands.add_parser(
        "evaluate", parents=[common],
        help="score Jev or the local model on the test set, asking questions the way a scan does",
    )
    check.add_argument("--file", help="dataset file to score (default: <data_dir>/dataset/test.jsonl)")
    check.add_argument("--backend", choices=BACKENDS, help="default: DECISION_BACKEND from .env, else jev")
    check.add_argument("--accept", type=float, help="accept threshold (default: from the config)")
    check.add_argument("--reject", type=float, help="reject threshold (default: from the config)")
    check.add_argument(
        "--workers", type=int, help="parallel requests (default: 10 for Jev, 2 for the local model)",
    )
    check.set_defaults(command=run_evaluate)

    stats = commands.add_parser("stats", parents=[common], help="show totals and recent scans")
    stats.set_defaults(command=run_stats)

    return parser


def run_scan(args, profile, db) -> int:
    source = ArxivSource()
    categories = args.category or profile.categories
    if args.since == "last-run":
        since = since_last_run(db, source.name, categories)
    else:
        since = date.fromisoformat(args.since)
    until = date.fromisoformat(args.until) if args.until else None

    log.info(
        "Scanning %s for '%s' since %s with %s",
        ", ".join(categories), profile.topic, since, backend_name(args.backend),
    )
    jev = JevClient(backend=args.backend)
    try:
        workers = args.workers or DEFAULT_WORKERS[jev.backend]
        pipeline = Pipeline(profile, db, jev, source, workers=workers)
        summary = pipeline.scan(since, until, categories, max_papers=args.max)
    finally:
        jev.close()

    export_accepted(db, profile.obsidian_dir)
    log.info(
        "Done: %d new papers, %d accepted, %d maybe, %d rejected, %d failed, cost $%.4f",
        summary.new, summary.accepted, summary.maybe, summary.rejected, summary.failed, summary.cost_usd,
    )
    for title in summary.accepted_titles:
        log.info("  accepted: %s", title)

    if summary.accepted and not args.no_notify:
        notify(
            f"{profile.topic}: {summary.accepted} new paper(s)",
            "\n".join(summary.accepted_titles[:3]),
        )
    return 0


def run_download_all(args, profile, db) -> int:
    statuses = [status.strip() for status in args.status.split(",")]
    count = len(db.papers_missing_pdf(statuses))
    log.info("%d PDFs to download into %s (about %d min)", count, profile.pdf_dir, count * 4 // 60 + 1)
    downloaded = download_all(db, profile.pdf_dir, statuses)
    log.info("Downloaded %d of %d PDFs", downloaded, count)
    return 0


def run_export(args, profile, db) -> int:
    count = export_accepted(db, profile.obsidian_dir)
    log.info("Wrote %d notes to %s", count, profile.obsidian_dir)
    return 0


def run_export_dataset(args, profile, db) -> int:
    out_dir = args.out or profile.dataset_dir
    summary = export_dataset(db, profile, out_dir, test_size=args.test_size)
    for split in ("train", "test"):
        log.info(
            "%s.jsonl: %d papers, %d relevant", split, summary.rows[split], summary.relevant[split]
        )
    log.info(
        "Gold relevance from your labels: %d, from Jev: %d. Skipped %d unlabelled maybe papers.",
        summary.label_sources["human"], summary.label_sources["jev"], summary.skipped_maybe,
    )
    if summary.rows["test"]:
        log.info(
            "Test answers from your labels: %d, from Jev: %d",
            summary.test_sources["human"], summary.test_sources["jev"],
        )
    else:
        log.warning(
            "The test set is empty. Add --test-size 300, or label papers in the dashboard's "
            "Label tab (test sample)."
        )
    log.info("Wrote %s", out_dir)

    if args.push is not None:
        repo = args.push or os.getenv("HF_DATASET_REPO")
        if not repo:
            raise RuntimeError("--push needs a repository: set HF_DATASET_REPO in .env, or use --push USER/REPO")
        url = push_dataset(out_dir, repo, public=args.public)
        log.info("Pushed to %s", url)
    return 0


def run_labels_export(args, profile, db) -> int:
    path = args.out or profile.labels_path
    count = write_labels_file(db, path)
    log.info("Wrote %d labels to %s", count, path)
    return 0


def run_labels_import(args, profile, db) -> int:
    path = Path(args.file or profile.labels_path)
    if not path.is_file():
        raise RuntimeError(
            f"No labels file at {path}. Pass the path of your backup, e.g. "
            f"{profile.labels_path} (from 'labels export') or {profile.dataset_dir / 'labels.jsonl'} "
            "(from 'export-dataset')."
        )
    counts = db.import_labels(read_labels_file(path))
    log.info(
        "Labels: %d added, %d updated, %d already up to date",
        counts["added"], counts["updated"], counts["unchanged"],
    )
    known = set(db.get_labels()) & {paper["paper_id"] for paper in db.get_papers()}
    waiting = len(db.get_labels()) - len(known)
    if waiting:
        log.info("%d labels are for papers not scanned yet; they apply once you scan them.", waiting)
    return 0


def run_evaluate(args, profile, db) -> int:
    path = Path(args.file or profile.dataset_dir / "test.jsonl")
    if not path.is_file():
        raise RuntimeError(f"No dataset file at {path}. Run 'export-dataset --test-size 300' first.")
    stage = relevance_stage(profile)
    if args.accept is not None:
        stage.accept_threshold = args.accept
    if args.reject is not None:
        stage.reject_threshold = args.reject

    backend = backend_name(args.backend)
    log.info(
        "Scoring %s on %s (accept >= %.2f, reject <= %.2f)",
        backend, path, stage.accept_threshold, stage.reject_threshold,
    )
    client = JevClient(backend=backend)
    try:
        report = evaluate(profile, client, path, workers=args.workers or DEFAULT_WORKERS[backend])
    finally:
        client.close()

    def percent(value):
        return "  -   " if value is None else f"{value:6.1%}"

    print(f"\n{'answers from':<14}{'papers':>7}{'relevant':>9}  {'found':>6}  {'precision':>9}  {'maybe':>6}  {'sub-topics':>10}")
    for source, name in ((None, "all"), ("jev", "Jev"), ("human", "your labels")):
        scores = report.scores(source)
        if scores["papers"]:
            print(
                f"{name:<14}{scores['papers']:>7}{scores['relevant']:>9}  {percent(scores['found'])}  "
                f"{percent(scores['precision']):>9}  {percent(scores['maybe'])}  {percent(scores['sub_topics']):>10}"
            )
    for title, papers in (("Missed relevant papers", report.missed()), ("Wrongly accepted", report.false_accepts())):
        if papers:
            print(f"\n{title} ({len(papers)}):")
            for paper in papers[:10]:
                print(f"  {paper.probability:.2f}  {paper.paper_id}  {paper.title[:80]}")
    print(f"\n{report.input_tokens:,} input tokens, cost ${report.cost_usd:.4f}")
    return 0


def run_stats(args, profile, db) -> int:
    totals = db.totals()
    print(f"Topic:     {profile.topic}")
    print(f"Database:  {profile.db_path}")
    print(f"Papers:    {totals['papers']}  (accepted {totals['accepted']}, maybe {totals['maybe']}, "
          f"rejected {totals['rejected']}, pending {totals['pending']})")
    print(f"Jev calls: {totals['jev_calls']}, {totals['input_tokens']:,} input tokens, "
          f"cost ${totals['cost_usd']:.4f}")
    labels = db.label_counts()
    print(f"Labels:    {labels['train']} train, {labels['test']} test, "
          f"{labels['maybe_left']} maybe papers still unlabelled")
    print("\nRecent scans:")
    for run in db.get_runs(limit=10):
        params = run["query_params"]
        print(f"  #{run['run_id']:<4} {run['started_at'][:16]}  {run['status']:<11} "
              f"{','.join(params['categories'])} {params['since']} .. {params['until']}  "
              f"checkpoint {run['checkpoint'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
