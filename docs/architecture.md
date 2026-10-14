# ADR-001: Architecture review of the implementation plan

**Status:** Accepted
**Date:** 2026-09-24
**Deciders:** project owner

## Context

`.agent/PLAN.md` describes the tool: arXiv harvesting, Jev classification in config-driven stages,
SQLite storage, a Streamlit dashboard, cron scheduling, PDF downloads, a paper graph and an
Obsidian export. Before implementing it, the plan was checked against the current Jev
documentation (docs.typesafe.ai) and against the live arXiv services. The tool is for a single user
on a single machine, will be published as open source, and has to handle both backfills of tens of
thousands of papers and small daily runs.

## Decisions kept from the plan

- **SQLite as the source of truth, plus a Markdown export.** Zero setup, one file, easy to back up.
  The export covers reading on other devices without a cloud database.
- **Stages in YAML instead of a workflow engine.** Two or three sequential questions don't justify
  running n8n or Langflow next to the tool.
- **One Jev call per stage.** Jev's docs recommend packing many questions into one call, but here
  most papers are rejected by the first question. Stopping at that point means the later questions
  are only paid for on the few papers that pass, which is cheaper than asking everything up front.
- **Accept / maybe / reject thresholds on the relevance question.** This matches the three-way
  routing TypeSafe recommends for Noul answers (their example uses 0.8 and 0.2).
- **Streamlit, cron / Task Scheduler, `PaperSource` interface, NetworkX + Pyvis.** All reasonable
  for the scope.

## Decisions changed during implementation

### 1. arXiv OAI-PMH instead of the search API

The plan assumed `export.arxiv.org/api/query`. In September 2026 that endpoint answers HTTP 406 to
most automated date-range queries. This was reproduced from the development machine, and other
projects on GitHub report the same thing. `oaipmh.arxiv.org` (OAI-PMH) worked and is arXiv's
official bulk harvesting interface. It supports per-category sets (`cs:cs:CV`) and from/until
dates, and it pages with resumption tokens.

Details:

- The `arXivRaw` metadata format is used because it lists every version with its date. The
  `created` field of the plain `arXiv` format turned out to hold the latest version's date.
- OAI dates are last-modified dates, so a harvest also contains revisions of old papers. Papers
  whose first version is more than 30 days older than the scan start are skipped.
- Scans run one day at a time. The checkpoint is the last finished day, which is simpler and
  sturdier than storing resumption tokens, since those expire after about a day.
- PDFs come from `arxiv.org/pdf/<id>`. `export.arxiv.org/pdf` also returned 406.

### 2. The official `typesafe-sdk` instead of a hand-written HTTP client

The SDK handles authentication, retries on 429 and 5xx with backoff, `Retry-After`, and typed
answers. `jev_client.py` is a thin adapter: stage config to question, answer to result dict, usage
to cost. Tests use the SDK with a mock HTTP transport. The environment variable is the SDK's
`TYPESAFE_API_KEY`, not `JEV_API_KEY`.

### 3. Jev pricing has a known default

Pricing is now public: $0.042 per million input tokens for jev-1.13, with output free. That is the
default, and `JEV_PRICE_PER_MILLION_INPUT_TOKENS` overrides it. Cost is computed from input tokens
only.

### 4. The graph has three selectable modes

With one main category per paper, "edge when two papers share a tag" connects every paper in a
category to every other. For 500 papers that is about 125,000 edges, which Pyvis can't draw
usefully. The dashboard offers three modes instead:

- tag hubs, where papers link to tag nodes and the edge count grows linearly
- similar papers, where each paper links to its k most similar papers by tag-probability overlap
- tag hubs plus shared-author edges

Tags are the selected option of each choice stage plus every option with probability of at least
`tag_threshold`, so papers that span two areas connect both clusters.

### 5. One topic per config file and data folder

Changing the topic in the dashboard would silently skip every paper already in the database,
because of the "never reprocess" rule. The topic therefore lives in the YAML file, the dashboard
shows it read-only and lets you pick a config file, and each config has its own `data_dir`.

### 6. Smaller changes

- `papers.paper_id` and `papers.source` replace `arxiv_id`, so a second source does not need a
  schema change.
- `papers` also stores the relevance probability and tags. They are denormalised from
  `stage_results` so the dashboard can filter without joins.
- Choice stages never reject. Low confidence only marks that stage as "maybe"; the paper's final
  status comes from the yes/no stages.
- All Jev calls for one paper run in a worker thread, and the results are written in one
  transaction from the main thread. A crash never leaves a half-classified paper; it stays
  `pending` and is retried on the next scan.
- Desktop notifications go through `plyer` and are best effort, so a missing backend never fails a
  scan.

### 7. Labels and a training dataset for a local model

Jev is a paid API. To replace it with a free local decision model, the database also keeps your
own relevance labels in a `labels` table (schema version 2; older databases gain the table when
opened).

- Jev's accept and reject decisions are the default gold answers, so a scan builds a dataset at
  no extra effort. Your label overrides Jev's, which is how "maybe" papers and Jev's mistakes get
  an answer.
- Each label has a split. Papers labelled from the dashboard's test sample are `test` and are
  never written to `train.jsonl`. The sample is balanced across Jev's accepted, maybe and rejected
  papers, because a plain random sample would hold almost no relevant papers and could not
  measure how many a model misses. Jev's answer is hidden while labelling it.
- `export-dataset --test-size N` can fill the test set without labelling, by holding out random
  papers with Jev's answers, half relevant (at most half of all relevant papers). It then
  measures agreement with Jev rather than correctness, which is enough to decide whether a local
  model can replace Jev. The held-out ids go into a `test_holdout` table (schema version 3)
  instead of being re-drawn on each export: a re-draw over a growing set of papers would move
  some tested papers into training.
- Labels answer the relevance question only. Later stages take Jev's answers as gold and are
  only asked for relevant papers, because their questions assume an on-topic paper.
- The export uses the row format of Unsloth's decision-model trainer (`state`, `questions`,
  `gold` as JSON strings), and the state is built by the same function the scan uses, so a model
  trained on it sees exactly what Jev saw.

### 8. A local decision model as a second backend

The labels and the export (section 7) exist to train a free replacement for Jev. The rest of the
tool does not change for it:

- **Same API.** llama.cpp's server answers TypeSafe's System One requests at `/v1/systemone`, so
  `JevClient` talks to it through the same `typesafe-sdk`, only with another base URL and a
  placeholder key. `DECISION_BACKEND=local` (or `--backend local`) selects it; its cost is
  recorded as $0 and the `model` column says which model answered. A scan checks `/health` first,
  so a stopped server fails once with the start command instead of failing every paper.
- **CPU-friendly defaults.** A laptop CPU needs about a second per answer, more while requests
  share it, so the local backend waits up to `LOCAL_DECISION_TIMEOUT` (120 s) and sends 2
  parallel requests instead of 10.
- **One question per request, in training as in scanning.** Clef heads (Unsloth's decision head
  on Qwen) read all questions of a request together. In the dataset, only relevant papers have
  sub-topic questions, so a model trained or tested with all of a row's questions at once can
  tell relevance from the number of questions. The first trained model did exactly that: perfect
  in its notebook, 0% found in a real scan. The notebook now splits rows into one question each,
  and `evaluate` asks the way `Pipeline.classify` does: the relevance question alone, then each
  later stage on its own.
- **`evaluate` decides whether a model is good enough.** It scores any backend on `test.jsonl`
  (found, precision, maybe, sub-topics; separately for Jev's answers and your labels) and shows
  the same numbers at other thresholds from the answers it already has, so thresholds are chosen
  from a table rather than tuned to a handful of test papers.
- **Training stays out of the repository's runtime.** The Colab notebook trains with Unsloth on a
  free GPU and uploads a GGUF; the scanner only needs llama.cpp to serve it. Results of the runs
  so far are in [training-results.md](training-results.md).

## Consequences

- Getting started is easy: install, add a key, run. There is nothing to host.
- Big backfills are limited by arXiv's request rate, not by Jev. A year of one category takes
  hours, but it can be interrupted and resumed at any time.
- Changing a topic's questions does not re-classify old papers. Start a new data folder instead.
- Keyword search is not available because OAI-PMH has none. Jev is the filter.
- A local model learns Jev's boundary from Jev's own answers plus a few hand labels. It copies
  Jev's clear answers well but is overconfident on the borderline papers Jev left at "maybe";
  labelling those papers matters more than scanning more.

## Follow-ups

1. Run a first real scan with an API key and tune the thresholds on real answers.
2. Pin the model version (`TYPESAFE_DEFAULT_MODEL=jev-1.13.0`) once the thresholds are tuned,
   because the `jev-latest` alias can move.
3. Later: a second pass over full PDF text for richer paper-to-paper links.
