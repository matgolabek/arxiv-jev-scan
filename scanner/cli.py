"""Command line interface. Run `python -m scanner.cli --help`."""

import argparse
import logging
import sys
from datetime import date

from dotenv import load_dotenv

from scanner.config import load_profile
from scanner.db import Database
from scanner.exporters.dataset import export_dataset
from scanner.exporters.obsidian import export_accepted
from scanner.jev_client import JevClient
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
    scan.add_argument("--workers", type=int, default=10, help="parallel Jev calls (default: %(default)s)")
    scan.add_argument("--no-notify", action="store_true", help="skip the desktop notification")
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
    dataset.set_defaults(command=run_export_dataset)

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

    log.info("Scanning %s for '%s' since %s", ", ".join(categories), profile.topic, since)
    jev = JevClient()
    try:
        pipeline = Pipeline(profile, db, jev, source, workers=args.workers)
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
    summary = export_dataset(db, profile, out_dir)
    for split in ("train", "test"):
        log.info(
            "%s.jsonl: %d papers, %d relevant", split, summary.rows[split], summary.relevant[split]
        )
    log.info(
        "Gold relevance from your labels: %d, from Jev: %d. Skipped %d unlabelled maybe papers.",
        summary.label_sources["human"], summary.label_sources["jev"], summary.skipped_maybe,
    )
    if summary.rows["test"] == 0:
        log.warning("The test set is empty. Label papers in the dashboard's Label tab (test sample).")
    log.info("Wrote %s", out_dir)
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
