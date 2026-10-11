# Replacing Jev with a local model: first training run

**Result: a fine-tuned Qwen3.5-0.8B agreed with Jev on all 300 test papers.** It found every
relevant paper, accepted no irrelevant one and left nothing at "maybe". It runs locally for free and
is published as a GGUF file for llama.cpp:
[matgolabek/arxiv-event-cameras-model](https://huggingface.co/matgolabek/arxiv-event-cameras-model).

| | |
|---|---|
| Topic | Event cameras (`config/stages.yaml`) |
| Dataset | [matgolabek/arxiv-event-cameras](https://huggingface.co/datasets/matgolabek/arxiv-event-cameras) |
| Notebook | [`notebooks/train_decision_model.ipynb`](../notebooks/train_decision_model.ipynb) |
| Hardware | Google Colab, free Tesla T4 (15 GB) |
| Software | Unsloth 2026.10.3, Transformers 5.17.0, Torch 2.11.0 |
| Date | October 2026 |

## The data

About 40,000 arXiv papers from cs.CV and cs.RO, classified by Jev. Fewer than 1% are about
event cameras.

| Split | Papers | Relevant | Source of the answers |
|---|---:|---:|---|
| `train.jsonl` | 39,458 | 177 | Jev, plus 22 of your own labels |
| `test.jsonl` | 300 | 150 | Jev for 283 papers (`--test-size 300`), your labels for 17 |

The models did not train on all 39,458 papers, because a model that sees 99% "no" learns to say no.
They trained on a **balanced subset of 1,947 papers**: all 177 relevant ones plus 1,770 rejected ones.
Half of those rejected papers were the ones Jev found most borderline, the other half random.

## Results

Test set: 300 papers, half of them relevant. Thresholds are the scanner's defaults (accept at a
probability of at least 0.8, reject at most 0.2, "maybe" in between).

| Model | Relevant papers found | Precision of accepts | Left at "maybe" | Sub-topic matches Jev |
|---|---:|---:|---:|---:|
| Laya, zero-shot | 100% | 70.4% | 46.0% | 19.3% |
| Laya, fine-tuned | 80.0% | 100% | 5.0% | 68.0% |
| **Qwen3.5-0.8B, fine-tuned** | **100%** | **100%** | **0.0%** | **84.7%** |

- **Relevant papers found**: share of relevant papers not rejected. Missing a paper is the costly
  error, so this matters most.
- **Precision of accepts**: share of accepted papers that are relevant.
- **Left at "maybe"**: share of all papers left for you to review.
- **Sub-topic matches Jev**: for relevant papers, how often the `domain` answer (Robotics,
  Perception, ...) matches Jev's.

With thresholds tuned on the test set (find at least 95% of relevant papers, at least 90% of accepts
relevant, reject at most 0.5):

| Model | Accept / reject | Found | Precision | Maybe |
|---|---|---:|---:|---:|
| Laya, zero-shot | 0.87 / 0.50 | 100% | 90.0% | 73.0% |
| Laya, fine-tuned | 0.11 / 0.01 | 94.0% | 100% | 16.3% |
| **Qwen3.5-0.8B, fine-tuned** | **0.60 / 0.50** | **100%** | **100%** | **0.0%** |

Qwen's scores fall cleanly on both sides of 0.5, so the tuned thresholds only add a margin. The
suggestion is capped at 0.5 for reject and 0.1 above it for accept.

### Training details

| | Laya, fine-tuned | Qwen3.5-0.8B, fine-tuned |
|---|---|---|
| Base model | `convaiinnovations/laya` (ModernBERT-large, ~0.4B) | `unsloth/Qwen3.5-0.8B` + Clef decision head |
| Trainable parameters | LoRA r=32 | 49.0M of 902M (5.4%), LoRA r=32 |
| Training decisions | 2,124 | 1,947 |
| Context | 512 tokens (118 inputs cut short) | 1,024 tokens |
| Steps | 2 epochs | 220 (2 epochs, batch 16) |
| Holdout accuracy after calibration | 95.8% (ECE 0.027) | 99.5% (ECE 0.011) |

## What these numbers do and don't show

- **They measure agreement with Jev.** 283 of the 300 test answers are Jev's own. Jev scores 100% on
  them by definition, so this shows that Qwen can stand in for Jev, not that either is always right.
  The 17 papers you labelled yourself were all ones Jev called "maybe", and Qwen got them right too.
- **False accepts are not measured yet.** The test set holds only 150 irrelevant papers, too few to
  show a false-accept rate under about 0.7%. Real scans are over 99% irrelevant, so even 0.5% would
  add about 5 wrong papers per 1,000 scanned, next to about 8 relevant ones. This run used the
  notebook version without that check. The current notebook also scores each model on 2,000
  rejected papers it never trained on, and prints false accepts per 1,000 scanned papers.
- **Laya was cut short.** 118 of its 2,124 training inputs were longer than its 512-token window.
  That probably explains part of its lower recall, but it would still trail Qwen.
- **Sub-topics are the weak spot.** Qwen matched Jev's `domain` answer for 84.7% of relevant papers.
  Many of these papers fit two areas, and tags use every option above the tag threshold anyway, so
  this matters less than relevance.

## Next steps

1. Re-run the current notebook to get the false-accept rate on unseen rejected papers.
2. Add a local backend to the scanner: llama.cpp's decision server (`/v1/systemone`) serving this
   GGUF, selected in `.env`, with $0 cost.
3. Add an `evaluate` command that runs any backend on `test.jsonl`.
4. Use `accept_threshold: 0.60` and `reject_threshold: 0.50` for the relevance stage when the
   scanner runs on the local model, and run it next to Jev for a week or two before switching off
   Jev.
