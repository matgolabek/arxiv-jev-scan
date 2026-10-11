# arxiv-jev-scan

Keep up with arXiv on one specific topic without reading hundreds of abstracts a day.

You describe the topic in a small YAML file, for example "papers about event cameras, and not about
events in the everyday sense". The scanner harvests new papers from the arXiv categories you choose
and asks [Jev](https://docs.typesafe.ai), TypeSafe's classification model, whether each paper
matches. Papers that match are sorted into sub-topics. You browse the results in a small web
dashboard, download the PDFs you care about, and get a desktop notification when something new
shows up.

It is built for one person on one machine: SQLite for storage, Markdown notes for syncing to your
other devices, and no servers or accounts beyond a Jev API key.

## What it does

- Scans any arXiv categories (cs.CV, cs.RO, eess.IV, ...), from a one-off backfill of a whole
  year down to a quick daily update.
- Never sends the same paper to Jev twice, no matter how often you re-run a scan.
- Classifies in stages defined in YAML. The first stage is a yes/no relevance question with an
  accept / maybe / reject band, so borderline papers are kept for you to review instead of being
  dropped. Later stages only run for papers that passed, so a rejected paper costs a single call.
- Resumes an interrupted backfill from the last finished day.
- Shows every paper's per-stage answers, token usage and cost in the dashboard.
- Downloads PDFs one at a time or all at once, with polite rate limiting.
- Draws a network graph of related papers.
- Exports accepted papers as Markdown notes with YAML frontmatter, ready for Obsidian.

## Requirements

- Python 3.10 or newer
- A TypeSafe API key from https://console.typesafe.ai/keys

## Installation

```bash
git clone https://github.com/<you>/arxiv-jev-scan.git
cd arxiv-jev-scan

python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your API key in .env
```

## Describe your topic

Everything about a topic lives in `config/stages.yaml`. The file that ships with the project looks
for event camera papers:

```yaml
topic: Event cameras
data_dir: data

source:
  categories: [cs.CV, cs.RO]

stages:
  - name: relevance
    type: noul                 # a yes/no question
    instructions: >-
      Is this paper about event cameras, meaning neuromorphic vision sensors
      that report per-pixel brightness changes, or about processing the event
      streams they produce?
    criteria:
      true: The paper uses, studies, simulates or builds event cameras ...
      false: The paper does not involve event cameras. Event detection in ordinary video ...
    accept_threshold: 0.8      # probability >= 0.8 is accepted
    reject_threshold: 0.2      # probability <= 0.2 is rejected, anything in between is "maybe"

  - name: domain
    type: choice               # pick one option; likely options become tags
    instructions: What is the main application area of this event camera paper?
    criteria:
      Robotics: Robots, drones, autonomous driving, navigation or control
      Perception: Object detection, tracking, recognition or segmentation
      Imaging: Video or image reconstruction, deblurring, HDR ...
      Other: None of the areas above
```

Jev reads questions quite literally, so spell out what counts as yes and what counts as no. The
comments in the file explain each field.

Each paper is stored once and never classified again, so one config file means one topic. To follow
a second topic, copy the file, change `topic`, `data_dir` and the stages, and pass it with
`--config`. The dashboard lets you switch between config files.

## Usage

### A first try

Scan a few days, but stop after 20 new papers:

```bash
python -m scanner.cli scan --since 2026-09-20 --max 20
```

The output looks like this:

```
10:02:11 INFO    Scanning cs.CV, cs.RO for 'Event cameras' since 2026-09-20
10:02:21 INFO    arXiv 2026-09-20: 212 records in cs.CV, cs.RO
10:02:25 INFO    Reached the limit of 20 new papers, stopping
10:02:25 INFO    Done: 20 new papers, 1 accepted, 0 maybe, 19 rejected, 0 failed, cost $0.0003
10:02:25 INFO      accepted: Multi-viewpoint Geo-localization with Event Cameras
```

### A big backfill

```bash
python -m scanner.cli scan --since 2026-01-01
```

The scanner works through the date range one day at a time and saves a checkpoint after each day.
If you stop it with Ctrl+C, or your laptop goes to sleep, run the same command again and it picks
up where it stopped. Harvesting is limited by arXiv's rules (one request every three seconds), so a
year of cs.CV takes a while. Leave it running in a terminal.

### Daily updates

```bash
python -m scanner.cli scan
```

Without `--since`, the scan starts at the last day a previous scan finished. Papers already in the
database are skipped before any Jev call, so overlapping runs cost nothing.

### The dashboard

```bash
streamlit run app.py
```

This opens http://localhost:8501 in your browser. From there you can:

- start a scan from the sidebar, either "since the last scan" or for a date range
- see totals for papers, tokens and cost
- browse the Relevant, Maybe and Rejected tabs, and click a row to see the abstract, each stage's
  decision, the relevance probability and the sub-topic probabilities
- download a single PDF, or all of them with "Download all PDFs"
- open the Graph tab, where you choose how papers are connected: through shared tag hubs, directly
  to their most similar papers, or through tag hubs plus shared authors

### PDFs and notes

```bash
python -m scanner.cli download-all                    # PDFs of accepted papers
python -m scanner.cli download-all --status accepted,maybe
python -m scanner.cli export                          # Obsidian notes, also done after every scan
python -m scanner.cli stats                           # totals, label counts and recent scans
python -m scanner.cli labels export                   # back up your labels
```

`download-all` skips papers whose PDF is already downloaded, so running it again only fetches what
is missing.

### Labelling papers and building a training dataset

This fork aims to replace Jev with a free local decision model (Laya, or a Qwen model fine-tuned
with [Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth), for
example on a free Colab GPU). The first step is a dataset: Jev's answers serve as the teacher, and
your own labels fix the cases it was unsure about or got wrong.

In the dashboard, the **Label** tab has two queues:

- **Maybe papers.** Papers Jev was unsure about, most likely relevant first. Answer Relevant or
  Not relevant. These become training data. Maybe papers you don't label are left out of the
  dataset, since nobody knows their answer.
- **Test sample.** Random papers, balanced between those Jev accepted, rejected and was unsure
  about, with Jev's answer hidden. They are held out and never trained on, so they show how a
  local model compares with Jev. A few hundred is a good size.

You can also label any paper from its details in the Relevant, Maybe and Rejected tabs, for
example when you spot a mistake. The "You" column shows your labels.

Then export the dataset:

```bash
python -m scanner.cli export-dataset --test-size 300    # writes data/dataset/
```

`--test-size 300` holds out random papers, half relevant and half not, with Jev's answers as
gold, until the test set has 300 papers. Papers you labelled from the test sample count towards
the 300. At most half of the relevant papers are held out, so training keeps enough of them. The
held-out papers are remembered in the database, so later exports keep them in the test set even
without the option and after new scans; a paper that was tested on never moves into training.

A test set with Jev's answers measures how closely a model copies Jev, and Jev itself scores 100%
on it. That is enough to decide whether a local model can replace Jev. To also measure how often
Jev and the model are actually right, label papers from the test sample yourself; each test row's
`label_source` says where its answer came from.

```
data/dataset/
  train.jsonl      every paper with a known answer, except the test set
  test.jsonl       held-out papers and the papers you labelled from the test sample
  manifest.json    row counts, relevant vs not, where the labels came from
  labels.jsonl     a backup of your labels (see below)
  README.md        dataset card for the Hugging Face Hub
```

Each line uses the row format of Unsloth's decision-model trainer: `state` (title and abstract),
`questions` (the stages from your config) and `gold` (the answers), each as a JSON string. The
relevance answer is your label when you gave one, otherwise Jev's accept or reject. Later stages
are only asked for papers that are relevant, with Jev's answers as gold; choice stages keep Jev's
probabilities as soft labels. Each line also carries `jev` with Jev's raw answers, for comparing
models later, plus `id`, `split`, `label_source` (`human` or `jev`) and `published`.

Relevant papers are rare, so scan enough days to have at least a hundred or so of them before
training. Before publishing a model trained on Jev's answers, check TypeSafe's terms of service.

#### Sharing the dataset through the Hugging Face Hub

Keep the dataset out of git: every export rewrites the files, so each one would stay in the
history forever, and two branches with different exports can't be merged. A Hugging Face dataset
repository versions every upload and loads straight into Colab. Log in once
(`hf auth login`, with a token that can write), put the repository name in `.env`:

```
HF_DATASET_REPO=your-name/arxiv-event-cameras
HF_MODEL_REPO=your-name/arxiv-event-cameras-model
```

then export and upload in one go:

```bash
python -m scanner.cli export-dataset --test-size 300 --push
```

`--push your-name/other-repo` uploads somewhere else instead. The repository is created as private
unless you add `--public`.

#### Training a local model in Colab

[`notebooks/train_decision_model.ipynb`](notebooks/train_decision_model.ipynb) runs on a free Colab
T4 GPU. Open it in Colab (File > Open notebook > GitHub, or upload it), choose a T4 runtime, and drag
your `.env` into the Files pane, or add `HF_DATASET_REPO`, `HF_MODEL_REPO` and `HF_TOKEN` as Colab
secrets. It:

1. loads the dataset from `HF_DATASET_REPO` and builds a balanced training set: every relevant
   paper plus ten rejected ones each, half of them the rejected papers Jev found most borderline
2. scores zero-shot Laya on the test set
3. fine-tunes Laya and Qwen3.5-0.8B with [Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
   and scores them: the share of relevant papers found, the precision of accepted papers, the share
   left at "maybe", and sub-topic accuracy
4. suggests accept and reject thresholds and uploads the best model as GGUF to `HF_MODEL_REPO`

#### Backing up your labels

Your labels are the only part of this that costs your time, and they live in `papers.db`. Save
them to a small file, and load them back on another machine or into a fresh database:

```bash
python -m scanner.cli labels export                 # writes data/labels.jsonl
python -m scanner.cli labels import                 # reads data/labels.jsonl
python -m scanner.cli labels import path/to/labels.jsonl   # or any other backup
```

`export-dataset` also writes `labels.jsonl` into the dataset folder, so `--push` backs them up.
Importing keeps the newer label when both sides labelled a paper. Labels for papers you have not
scanned yet are kept and apply once a scan reaches them, so on a new machine you can import first
and scan afterwards.

## Where your data goes

```
data/
  papers.db        SQLite database: papers, every Jev answer, scan runs
  pdfs/            downloaded PDFs, named after the arXiv id
  obsidian/        one Markdown note per accepted paper
  dataset/         training dataset from `export-dataset`
```

`data/` is in `.gitignore`. To read your papers on another device, open `data/obsidian` as an
Obsidian vault, or sync it with git, Dropbox or Obsidian Sync.

## Running on a schedule

arXiv announces new papers on weekday evenings (US Eastern time), so a morning run picks up each
new batch.

Linux and macOS, with `crontab -e`:

```
0 8 * * * cd /path/to/arxiv-jev-scan && DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus .venv/bin/python -m scanner.cli scan >> data/scan.log 2>&1
```

The `DBUS_SESSION_BUS_ADDRESS` part is Linux-only. Cron jobs don't see your desktop session, and
without it the notification can't be shown. The scan runs either way. On macOS, notifications
through `plyer` need `pyobjus`; without it you only get the log.

Windows, in a terminal:

```
schtasks /Create /SC DAILY /ST 08:00 /TN arxiv-jev-scan ^
  /TR "cmd /c cd /d C:\path\to\arxiv-jev-scan && .venv\Scripts\python -m scanner.cli scan >> data\scan.log 2>&1"
```

## Costs

Jev charges for input tokens only. At the time of writing it costs $0.042 per million tokens for
`jev-1.13`, and output tokens are free. A title plus abstract plus the question comes to roughly
300 to 400 tokens, so classifying a thousand papers costs about one or two cents. The cost shown in
the dashboard uses the price set in `.env` (`JEV_PRICE_PER_MILLION_INPUT_TOKENS`). Change it there
if your price is different.

## How it works

1. **Harvest.** Papers come from arXiv's [OAI-PMH](https://info.arxiv.org/help/oa/index.html)
   interface, one category set and one day at a time. This is arXiv's official bulk metadata
   service. The search API (`export.arxiv.org/api/query`) has been rejecting most automated
   date-range queries with HTTP 406, so it isn't used. Revisions of old papers also show up in the
   harvest and are skipped.
2. **Dedup.** Every paper id is inserted into SQLite first. Only ids the database has never seen
   continue.
3. **Classify.** Each new paper goes through the stages in order, one Jev call per stage, with ten
   papers in parallel. The first `reject` stops the paper. The answers, tokens and cost of every
   call are stored.
4. **Checkpoint.** After each finished day the scan records that day, so an interrupted run
   continues from there.
5. **Export and notify.** Accepted papers are written as Markdown notes, and if any are new you get
   a desktop notification.

The design decisions and their trade-offs are written up in
[docs/architecture.md](docs/architecture.md).

## Project layout

```
config/stages.yaml        topic, categories and pipeline stages
scanner/
  sources/base.py         PaperSource interface
  sources/arxiv.py        arXiv OAI-PMH harvester
  jev_client.py           Jev calls through the official typesafe-sdk, cost accounting
  pipeline.py             dedup, stages, thresholds, checkpoints
  db.py                   SQLite schema and queries
  pdfs.py                 single and bulk PDF downloads
  exporters/obsidian.py   Markdown export
  exporters/dataset.py    training dataset export (train.jsonl, test.jsonl)
  notify.py               desktop notification
  cli.py                  command line interface
app.py                    Streamlit dashboard
graph_view.py             NetworkX + Pyvis graph
tests/                    pytest suite, no network or API key needed
```

## Tests

```bash
python -m pytest
```

The tests use canned arXiv responses and a mocked Jev API, so they run offline in about a second.

## Adding another source

Write a class that extends `PaperSource` in `scanner/sources/` and implements
`fetch_day(categories, day)`, returning a list of `Paper` objects. The pipeline, database and
dashboard don't need to change.

## Ideas for later

- A second pass that reads the full PDF text to find richer connections between papers than
  shared tags.
- More sources, such as OpenReview and bioRxiv.
