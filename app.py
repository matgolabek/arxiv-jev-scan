"""Streamlit dashboard. Start it with: streamlit run app.py"""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from graph_view import GRAPH_MODES, build_graph, tag_colors, to_html
from scanner.config import Profile, load_profile
from scanner.db import Database
from scanner.exporters.dataset import relevance_stage
from scanner.exporters.obsidian import export_accepted
from scanner.jev_client import JevClient, backend_name, local_url
from scanner.pdfs import download_all, download_pdf
from scanner.pipeline import Pipeline, since_last_run, utc_today
from scanner.sources import ArxivSource

DECISION_COLORS = {"accept": "green", "maybe": "orange", "reject": "red"}
LABEL_NAMES = {True: "yes", False: "no"}
MAX_GRAPH_PAPERS = 1500  # bigger graphs get slow in the browser

load_dotenv()
st.set_page_config(page_title="arxiv-jev-scan", layout="wide")


def main() -> None:
    profiles = sorted(str(path) for path in Path("config").glob("*.yaml"))
    profile = load_profile(st.sidebar.selectbox("Profile", profiles))
    db = Database(profile.db_path)

    st.title(profile.topic)
    st.caption(f"Relevance question: {profile.stages[0].instructions}")

    scan_controls(profile, db)
    recent_scans(db)

    totals = show_metrics(db)
    if totals["papers"] == 0:
        st.info("No papers yet. Pick a date range in the sidebar and run a scan.")
        return

    download_controls(profile, db)
    papers = pd.DataFrame(db.get_papers())
    labels = db.get_labels()
    papers["label"] = papers["paper_id"].map(
        lambda paper_id: LABEL_NAMES[labels[paper_id]["relevant"]] if paper_id in labels else ""
    )

    tabs = st.tabs(["Relevant", "Maybe", "Rejected", "Label", "Downloaded", "Graph"])
    with tabs[0]:
        paper_table(papers[papers.final_status == "accepted"], "relevant", profile, db)
    with tabs[1]:
        paper_table(papers[papers.final_status == "maybe"], "maybe", profile, db)
    with tabs[2]:
        paper_table(papers[papers.final_status == "rejected"], "rejected", profile, db)
    with tabs[3]:
        label_tab(profile, db)
    with tabs[4]:
        downloaded_table(papers)
    with tabs[5]:
        graph_tab(papers)


# --- sidebar ----------------------------------------------------------------


def scan_controls(profile: Profile, db: Database) -> None:
    sidebar = st.sidebar
    sidebar.header("Scan")
    text = sidebar.text_input("arXiv categories", ", ".join(profile.categories))
    categories = [category.strip() for category in text.split(",") if category.strip()]

    if sidebar.radio("Papers from", ["Since the last scan", "A date range"]) == "A date range":
        since = sidebar.date_input("From", utc_today() - timedelta(days=7))
        until = sidebar.date_input("To", utc_today())
    else:
        since = since_last_run(db, ArxivSource.name, categories)
        until = None
        sidebar.caption(f"Will scan from {since} to today.")

    max_papers = sidebar.number_input("Stop after N new papers (0 = no limit)", min_value=0, value=0, step=10)

    try:
        backend = backend_name()
        where = f" at {local_url()}" if backend == "local" else ""
        sidebar.caption(f"Classifier: **{backend}**{where} (DECISION_BACKEND in .env)")
    except RuntimeError as error:
        sidebar.error(str(error))
    if sidebar.button("Run scan", type="primary"):
        run_scan(profile, db, categories, since, until, max_papers or None)
    sidebar.caption("Big backfills are more comfortable from the command line, see the README.")


def run_scan(profile, db, categories, since, until, max_papers) -> None:
    try:
        jev = JevClient()
    except RuntimeError as error:
        st.error(str(error))
        return

    bar = st.progress(0.0, text="Starting scan...")
    pipeline = Pipeline(profile, db, jev, ArxivSource())
    try:
        summary = pipeline.scan(
            since, until, categories, max_papers,
            progress=lambda fraction, message: bar.progress(min(fraction, 1.0), text=message),
        )
    except Exception as error:
        st.error(f"Scan stopped: {error}. Run it again to continue where it left off.")
        return
    finally:
        jev.close()

    export_accepted(db, profile.obsidian_dir)
    st.success(
        f"{summary.new} new papers: {summary.accepted} accepted, {summary.maybe} maybe, "
        f"{summary.rejected} rejected. Cost ${summary.cost_usd:.4f}."
    )


def recent_scans(db: Database) -> None:
    runs = db.get_runs(limit=10)
    if not runs:
        return
    with st.sidebar.expander("Recent scans"):
        for run in runs:
            params = run["query_params"]
            st.caption(
                f"#{run['run_id']} {run['status']}: {', '.join(params['categories'])}, "
                f"{params['since']} to {params['until']} (done up to {run['checkpoint'] or '-'})"
            )


# --- main area --------------------------------------------------------------


def show_metrics(db: Database) -> dict:
    totals = db.totals()
    columns = st.columns(5)
    columns[0].metric("Papers scanned", f"{totals['papers']:,}")
    columns[1].metric("Accepted", f"{totals['accepted']:,}")
    columns[2].metric("Maybe", f"{totals['maybe']:,}")
    columns[3].metric("Tokens used", f"{totals['input_tokens'] + totals['output_tokens']:,}")
    columns[4].metric("Total cost", f"${totals['cost_usd']:.4f}")
    return totals


def download_controls(profile: Profile, db: Database) -> None:
    left, right = st.columns([1, 4])
    include_maybe = right.checkbox("Include maybe papers")
    if left.button("Download all PDFs"):
        statuses = ["accepted", "maybe"] if include_maybe else ["accepted"]
        bar = st.progress(0.0, text="Starting downloads...")
        count = download_all(
            db, profile.pdf_dir, statuses,
            progress=lambda fraction, message: bar.progress(min(fraction, 1.0), text=message),
        )
        st.success(f"Downloaded {count} PDFs into {profile.pdf_dir}")


def paper_table(papers: pd.DataFrame, key: str, profile: Profile, db: Database) -> None:
    if papers.empty:
        st.write("Nothing here yet.")
        return

    view = pd.DataFrame({
        "Title": papers["title"],
        "Published": papers["published_date"],
        "Relevance": papers["relevance"],
        "Tags": papers["tags"].apply(list),
        "You": papers["label"],
        "PDF": papers["pdf_path"].notna(),
        "Link": papers["url"],
    })
    event = st.dataframe(
        view,
        key=key,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Title": st.column_config.TextColumn(width="large"),
            "Relevance": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            "Tags": st.column_config.ListColumn(),
            "You": st.column_config.TextColumn(help="Your own relevance label, see the Label tab"),
            "PDF": st.column_config.CheckboxColumn(help="Already downloaded"),
            "Link": st.column_config.LinkColumn(display_text="arXiv"),
        },
    )

    if event.selection.rows:
        paper_details(papers.iloc[event.selection.rows[0]], key, profile, db)
    else:
        st.caption(f"{len(papers):,} papers. Select a row to see its details.")


def paper_details(paper: pd.Series, key: str, profile: Profile, db: Database) -> None:
    with st.container(border=True):
        st.subheader(paper["title"])
        st.caption(
            f"{', '.join(paper['authors'])}  |  first version {paper['published_date']}  |  {paper['paper_id']}"
        )
        text, stages = st.columns([3, 2])
        text.markdown(paper["abstract"])
        with stages:
            for result in db.get_stage_results(paper["paper_id"]):
                show_stage_result(result)
        label_buttons(paper["paper_id"], paper["label"], key, db)
        pdf_buttons(paper, key, profile, db)


def label_buttons(paper_id: str, current: str, key: str, db: Database) -> None:
    """Your own answer to the relevance question, used for the training dataset."""
    text, yes, no, clear, _ = st.columns([2, 1, 1, 1, 3], vertical_alignment="center")
    text.caption(f"Your label: **{current or 'none'}**")
    if yes.button("Relevant", key=f"label-yes-{key}"):
        db.set_label(paper_id, True)
        st.rerun()
    if no.button("Not relevant", key=f"label-no-{key}"):
        db.set_label(paper_id, False)
        st.rerun()
    if current and clear.button("Clear", key=f"label-clear-{key}"):
        db.remove_label(paper_id)
        st.rerun()


def show_stage_result(result: dict) -> None:
    st.badge(f"{result['stage_name']}: {result['decision']}", color=DECISION_COLORS[result["decision"]])
    answer = result["result"]
    if result["question_type"] == "noul":
        st.progress(answer["probability"], text=f"probability {answer['probability']:.2f}")
    else:
        st.caption(f"picked {answer['selected']} with confidence {answer['confidence']:.2f}")
        probabilities = pd.Series(answer["probabilities"], name="probability")
        st.bar_chart(probabilities, horizontal=True, height=40 + 25 * len(probabilities))


def pdf_buttons(paper: pd.Series, key: str, profile: Profile, db: Database) -> None:
    left, right, _ = st.columns([1, 1, 4])
    left.link_button("Open on arXiv", paper["url"])
    pdf_path = paper["pdf_path"]
    if isinstance(pdf_path, str) and Path(pdf_path).exists():
        right.download_button(
            "Save PDF", Path(pdf_path).read_bytes(), file_name=Path(pdf_path).name,
            mime="application/pdf", key=f"save-{key}",
        )
    elif right.button("Download PDF", key=f"download-{key}"):
        with st.spinner("Downloading from arXiv..."):
            download_pdf(db, paper.to_dict(), profile.pdf_dir)
        st.rerun()


# --- labelling -------------------------------------------------------------

LABEL_QUEUES = {
    "maybe": "Maybe papers (training data)",
    "test": "Test sample (held out for evaluation)",
}


def label_tab(profile: Profile, db: Database) -> None:
    counts = db.label_counts()
    by_status = counts["test_by_status"]
    columns = st.columns(3)
    columns[0].metric("Maybe papers left", f"{counts['maybe_left']:,}")
    columns[1].metric("Training labels", f"{counts['train']:,}")
    columns[2].metric(
        "Test labels", f"{counts['test']:,}",
        help=f"Drawn from Jev's accepted {by_status['accepted']}, maybe {by_status['maybe']}, "
             f"rejected {by_status['rejected']}",
    )

    queue = st.radio("Label", list(LABEL_QUEUES), format_func=LABEL_QUEUES.get, horizontal=True)
    if queue == "maybe":
        st.caption(
            "Papers Jev was unsure about, most likely relevant first. Your answers become "
            "training data; unlabelled maybe papers are left out of the dataset."
        )
    else:
        st.caption(
            "Random papers, balanced between those Jev accepted, rejected and was unsure about. "
            "They are never trained on, so they show how a local model compares with Jev. "
            "Jev's answer is hidden so it doesn't sway you. Aim for a few hundred."
        )

    # Keep the paper fixed across reruns until it is labelled or skipped.
    current_key, skip_key = f"label-current-{queue}", f"label-skipped-{queue}"
    skipped = st.session_state.setdefault(skip_key, [])
    paper = st.session_state.get(current_key)
    if paper is None or paper["paper_id"] in db.get_labels():
        if queue == "maybe":
            paper = db.next_unlabelled_maybe(skip=skipped)
        else:
            paper = db.next_test_candidate(skip=skipped)
        st.session_state[current_key] = paper
    if paper is None:
        st.success("Nothing left to label here.")
        if skipped and st.button("Show skipped papers again"):
            skipped.clear()
            st.rerun()
        return

    with st.container(border=True):
        st.subheader(paper["title"])
        st.caption(f"{paper['paper_id']}  |  first version {paper['published_date']}")
        st.markdown(paper["abstract"])
        if queue == "maybe":
            st.caption(f"Jev's relevance probability: {paper['relevance']:.2f}")
        st.caption(f"Question: {relevance_stage(profile).instructions}")

        yes, no, skip, link, _ = st.columns([1, 1, 1, 1, 2])
        split = "test" if queue == "test" else "train"
        if yes.button("Relevant", type="primary", key=f"queue-yes-{queue}"):
            db.set_label(paper["paper_id"], True, split)
            st.rerun()
        if no.button("Not relevant", type="primary", key=f"queue-no-{queue}"):
            db.set_label(paper["paper_id"], False, split)
            st.rerun()
        if skip.button("Skip", key=f"queue-skip-{queue}"):
            skipped.append(paper["paper_id"])
            st.session_state[current_key] = None
            st.rerun()
        link.link_button("Open on arXiv", paper["url"])

    st.caption(
        "When you have enough labels, run `python -m scanner.cli export-dataset` to write "
        f"train.jsonl and test.jsonl into {profile.dataset_dir}."
    )


def downloaded_table(papers: pd.DataFrame) -> None:
    have_pdf = papers[papers["pdf_path"].notna()]
    if have_pdf.empty:
        st.write("No PDFs yet. Use 'Download PDF' on a paper or 'Download all PDFs' above.")
        return
    view = pd.DataFrame({
        "Title": have_pdf["title"],
        "Status": have_pdf["final_status"],
        "File": have_pdf["pdf_path"],
        "Size (MB)": have_pdf["pdf_path"].apply(file_size_mb),
        "Downloaded": have_pdf["pdf_downloaded_at"],
        "Link": have_pdf["url"],
    })
    st.dataframe(
        view, hide_index=True,
        column_config={
            "Size (MB)": st.column_config.NumberColumn(format="%.1f"),
            "Link": st.column_config.LinkColumn(display_text="arXiv"),
        },
    )


def file_size_mb(path: str) -> float | None:
    file = Path(path)
    return file.stat().st_size / 1_000_000 if file.exists() else None


def graph_tab(papers: pd.DataFrame) -> None:
    candidates = papers[papers.final_status.isin(["accepted", "maybe"])]
    if candidates.empty:
        st.write("The graph shows accepted and maybe papers. There are none yet.")
        return

    left, middle, right = st.columns(3)
    mode = left.radio("Connect papers by", list(GRAPH_MODES), format_func=GRAPH_MODES.get)
    top_k = left.slider("Links per paper", 1, 10, 3) if mode == "similar" else 3
    statuses = middle.multiselect("Show", ["accepted", "maybe"], default=["accepted", "maybe"])
    min_relevance = middle.slider("Minimum relevance", 0.0, 1.0, 0.0, step=0.05)

    dates = pd.to_datetime(candidates["published_date"]).dt.date
    first, last = dates.min(), dates.max()
    if first < last:
        first, last = right.slider("First published", min_value=first, max_value=last, value=(first, last))

    selected = candidates[
        candidates.final_status.isin(statuses)
        & (candidates.relevance.fillna(0) >= min_relevance)
        & dates.between(first, last)
    ]
    if selected.empty:
        st.write("No papers match these filters.")
        return
    if len(selected) > MAX_GRAPH_PAPERS:
        st.caption(f"Showing the {MAX_GRAPH_PAPERS} most relevant of {len(selected):,} papers.")
        selected = selected.nlargest(MAX_GRAPH_PAPERS, "relevance")

    columns = ["paper_id", "title", "authors", "tags", "relevance", "final_status"]
    records_json = selected[columns].to_json(orient="records")
    st.iframe(graph_html(records_json, mode, top_k), height=720)
    show_legend(json.loads(records_json))
    st.caption("Circles are accepted papers, diamonds are maybe. Hover a node for details, scroll to zoom.")


@st.cache_data(show_spinner="Building graph...")
def graph_html(records_json: str, mode: str, top_k: int) -> str:
    # Takes JSON text so Streamlit can cache it (dict and list columns are not hashable).
    return to_html(build_graph(json.loads(records_json), mode, top_k))


def show_legend(papers: list[dict]) -> None:
    items = [
        f'<span style="color:{color}">&#9632;</span> {tag}'
        for tag, color in tag_colors(papers).items()
    ]
    st.markdown("&nbsp;&nbsp;".join(items), unsafe_allow_html=True)


main()
