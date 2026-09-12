# Local QLoRA Fine-Tuning for IT Support Triage

## Overview

This project fine-tunes [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with 4-bit QLoRA to triage IT support tickets: given raw ticket text, produce a structured JSON record with category, priority, a summary, troubleshooting steps, and an escalation recommendation. Training and all evaluation ran locally on a single consumer GPU.

The project runs two separate evaluations, not one. A 230-example closed-set benchmark shows near-perfect fine-tuned results, but the training loss curve suggested the model was largely memorizing repeated dataset templates rather than learning the task. A second, 75-example challenge set was then hand-written — deliberately avoiding those templates — to measure how much of that improvement actually holds up on unfamiliar tickets. That second result, a real but modest improvement, is the headline finding of this project, and the gap between the two is reported in full, including a specific case where fine-tuning made an escalation decision worse.

## Business Problem

IT support technicians spend real time on repetitive, low-judgment work before they reach the actual fix: reading a ticket, deciding which queue it belongs in, judging urgency, rewriting the issue into something scannable, and deciding whether it needs escalation. This project tests whether a locally fine-tuned LLM can produce a first-pass, structured suggestion for:

- **Categorization** — which of a fixed set of IT categories the ticket belongs to
- **Prioritization** — Low / Medium / High / Critical
- **Escalation** — whether and to whom the ticket should be escalated
- **Structured output** — a consistent JSON schema a downstream system could actually consume

## Hypothesis

**Alternative hypothesis (H1):** QLoRA fine-tuning improves adherence to an organization's ticket taxonomy, priority rules, and escalation conventions on *novel* tickets, compared to the base model given a well-engineered prompt — while needing a much smaller prompt at inference time.

**Null hypothesis (H0):** Fine-tuning provides no meaningful improvement over a well-prompted base model.

This is tested as two conditions, run against the identical set of tickets in every evaluation:

- **Condition A — Base + engineered prompt:** untouched Qwen2.5-7B-Instruct, given a detailed system prompt (`prompts/baseline_system_prompt.txt`) that spells out the JSON schema, valid categories, priority levels, and escalation targets, plus one worked example.
- **Condition B — Fine-tuned + minimal prompt:** the LoRA-adapted model, given only the bare instruction used during training (`"Analyze this IT support ticket." + ticket text`) — no schema description at all.

The engineered baseline matters because comparing fine-tuning against an *unprompted* base model would be a strawman — of course it would lose. Any gap that survives after fine-tuning represents value beyond what prompt engineering alone achieves, with a much smaller prompt.

**Result:** H1 is supported at the descriptive level on the 75-example novel challenge set (Category +4.0pp, Priority +8.0pp, Escalation +5.3pp), but this is a single run against a small, hand-built set — not a statistically validated claim (see [Limitations](#limitations)).

## Technical Approach

- **Base model:** Qwen2.5-7B-Instruct, loaded in 4-bit (`nf4`, double quantization, bfloat16 compute) via `bitsandbytes`
- **Fine-tuning:** LoRA via PEFT (`r=16, alpha=32, dropout=0.05`, applied to all attention and MLP projection matrices), trained with TRL's `SFTTrainer` using `assistant_only_loss=True` so only the JSON response — not the ticket text — contributes to the loss
- **Stack:** Python, PyTorch, Transformers, Datasets, PEFT, bitsandbytes, TRL, Accelerate
- **Hardware:** A single consumer GPU, 24GB VRAM
- **Environment:** Windows 11 + WSL2 (Ubuntu) — training and inference both ran entirely locally, no cloud GPU used

## Dataset

**All ticket data in this repository is synthetic**, generated for this project. It is not real corporate support tickets.

| Split | Examples | Purpose |
|---|---|---|
| Train | 1,120 | Fine-tuning |
| Validation | 150 | Training-time evaluation |
| Test (closed-set) | 230 | Frozen before evaluation; draws on the same generation templates as training |
| Challenge (novel) | 75 | Hand-authored afterward — no template reuse, heavy paraphrasing, typos, ambiguity, multi-symptom tickets |

The train/validation/test splits share the same underlying scenario templates (only details like department or app name vary between instances) — an efficient way to get a large, schema-consistent dataset, but one that lets a model score well by recognizing a template rather than reasoning about the ticket. That's what the closed-set results below reflect, which is why the challenge set exists.

## Experimental Design

Every result table below compares:

- **Condition A:** Base Qwen2.5-7B-Instruct + engineered system prompt
- **Condition B:** Fine-tuned Qwen2.5-7B-Instruct (LoRA) + minimal prompt

against the same fixed set of tickets, scored on JSON validity, required-field completion, category accuracy, priority accuracy, and escalation accuracy (`src/evaluate.py`).

## Closed-Set Results

**230 frozen, templated examples.**

| Metric | Base + Engineered Prompt | Fine-Tuned + Minimal Prompt |
|---|---|---|
| JSON validity | 97.4% | 100.0% |
| Required-field completion | 96.5% | 100.0% |
| Category accuracy | 87.0% | 99.6% |
| Priority accuracy | 57.8% | 100.0% |
| Escalation accuracy | 70.9% | 100.0% |

**These near-perfect fine-tuned numbers should not be read as real-world accuracy.** Training and validation loss collapsed almost immediately (validation loss dropped from ~1.17 to ~0.02 within the first epoch), which is consistent with the model recognizing which of a small set of scenario templates a ticket belongs to, rather than learning generalizable triage reasoning. The challenge-set results below are the more meaningful measure.

## Novel Challenge-Set Results

**This is the primary result of this project.** 75 examples, hand-authored after the closed-set was frozen, with typos, shorthand, incomplete sentences, multiple symptoms per ticket, irrelevant details, category-boundary cases, and tickets designed to test priority/escalation reasoning specifically (an executive's ticket that should stay Low priority, a security incident disguised as a mundane complaint, a VPN ticket where the VPN itself is fine).

| Metric | Base + Engineered Prompt | Fine-Tuned + Minimal Prompt | Change |
|---|---|---|---|
| JSON validity | 100.0% | 100.0% | +0.0pp |
| Required fields | 100.0% | 100.0% | +0.0pp |
| **Category accuracy** | **76.0%** | **80.0%** | **+4.0pp** |
| **Priority accuracy** | **77.3%** | **85.3%** | **+8.0pp** |
| **Escalation accuracy** | **82.7%** | **88.0%** | **+5.3pp** |

Compare category accuracy alone: 99.6% on the closed-set vs. 80.0% here, for the *same* fine-tuned model. That ~20-point gap is the actual measure of how much of the closed-set result was template recognition rather than task competence.

## Error Analysis

Because both conditions were run against the identical 75 challenge tickets, every example can be paired and classified. Net effect is positive on all three metrics, but fine-tuning was **not a strict improvement**:

| Metric | Both correct | Fine-tuned fixed | Fine-tuned regressed | Both wrong |
|---|---|---|---|---|
| Category | 50 | 10 | 7 | 8 |
| Priority | 54 | 10 | 4 | 7 |
| Escalation | 58 | 8 | 4 | 5 |

Two failure modes are worth calling out directly, not hidden:

- **Out-of-taxonomy category hallucination.** When the base model got a category wrong, its guesses stayed inside the valid 10-category list. When the fine-tuned model got a category wrong, it sometimes invented categories that don't exist in the schema at all (`Storage`, `Antivirus`, `Browser`, `Directory / Permissions`), and in one case output an escalation-target phrase (`"Identity and Access Management"`) as the *category* value.
- **A specific security-escalation regression.** One ransomware-pattern ticket (multiple users, replaced desktop backgrounds, scrambled filenames) was correctly escalated to `Security / Incident Response` by the base model, but the fine-tuned model misrouted it to `Desktop Support / Hardware`. More generally, **both** conditions occasionally under-escalated a should-escalate security ticket to `"No escalation needed"` — fine-tuning did not meaningfully fix this.

Full per-example detail: `evaluation/challenge_error_analysis.csv` and `evaluation/challenge_paired_analysis.csv`.

## Conclusion

QLoRA fine-tuning improved this model's average adherence to an organization-specific IT support taxonomy, priority rules, and escalation conventions on novel tickets, while needing a much smaller prompt than the best-effort baseline — a real but modest effect (+4 to +8 percentage points), not evidence of general IT-support reasoning. **It did not universally improve reliability**: it introduced a new failure mode (confidently inventing categories outside the valid taxonomy) and did not fix — and in one case worsened — a security-escalation decision. Production use of a system like this would require schema/taxonomy validation, a security-escalation guardrail independent of model output, and human review sitting between any prediction and a real consequence — not reliance on model output alone.

## Local Training Performance

On the local test hardware (a single consumer GPU, 24GB VRAM), full QLoRA training completed in ~7m 17s with ~9.28GB peak allocated VRAM — well under the available 24GB.

| | Value |
|---|---|
| GPU | Local consumer GPU, 24GB VRAM |
| Training time (3 epochs, 210 steps, 1,120 examples) | ~7m 17s (437.3s) |
| Peak training VRAM | ~9.28GB allocated |
| Adapter size | ~80.8MB (`adapter_model.safetensors`) |

## Repository Structure

```
data/            train/validation/test/challenge JSONL datasets
evaluation/      Frozen result CSVs (baseline, fine-tuned, error analysis, paired analysis)
models/          LoRA adapter config + training log (weights excluded from git, see below)
prompts/         Engineered baseline system prompt (exact text used)
src/             Training, evaluation, and analysis scripts
```

The LoRA adapter's binary weights (`adapter_model.safetensors`, ~80MB) are excluded from git to keep the repo small — regenerate them with `python src/train.py --output-dir models/it-support-lora` (~7 minutes on a single consumer GPU). The base model itself downloads from Hugging Face Hub on first run and is never stored in this repo.

## Setup / Usage

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Validate the datasets against the taxonomy
python src/validate_dataset.py --train data/train.jsonl --validation data/validation.jsonl --test data/test.jsonl
python src/validate_dataset.py --file data/challenge.jsonl

# Train (writes to models/it-support-lora/ — point --output-dir elsewhere to avoid overwriting)
python src/train.py --output-dir models/it-support-lora

# Evaluate: base model + engineered prompt
python src/evaluate.py --data data/test.jsonl \
    --system-prompt-file prompts/baseline_system_prompt.txt \
    --output evaluation/baseline_results.csv

# Evaluate: fine-tuned model + minimal prompt
python src/evaluate.py --data data/test.jsonl \
    --lora-path models/it-support-lora \
    --output evaluation/fine_tuned_results.csv

# Same two commands with --data data/challenge.jsonl and --output evaluation/challenge_*.csv
# reproduce the novel challenge-set results.

# Comparison table + error analysis
python src/analyze_results.py \
    --baseline evaluation/baseline_results.csv \
    --fine-tuned evaluation/fine_tuned_results.csv \
    --error-analysis-output evaluation/error_analysis.csv

python src/analyze_challenge.py   # paired analysis, reads evaluation/challenge_*_results.csv
```

## Limitations

- **All data is synthetic** — no real support tickets were used.
- **The challenge set is small (75 examples)** and single-author.
- **One base model, one training run, one random seed** — no hyperparameter sweep or seed variance estimate.
- **No formal statistical significance test** on the reported percentage-point improvements.
- **Not production-deployed** — evaluation is offline, against frozen datasets.
- **A real security-escalation regression was found** (see Error Analysis) — a limitation of the current adapter, not a hypothetical risk.
