# Finetuning Guide

## Quick Start

Minimal supervised finetune:

```bash
opf train /path/to/train.jsonl \
  --output-dir /path/to/finetuned_checkpoint
```

Use a dedicated validation split file (recommended):

```bash
opf train /path/to/train.jsonl \
  --validation-dataset /path/to/validation.jsonl \
  --output-dir /path/to/finetuned_checkpoint
```

## Expected Dataset Schema

`opf train` consumes the same dataset schema as `opf eval`:
- `text` (string)
- labeled spans in either `label` or `spans`

## Custom Label Spaces

Use `--label-space-json` to train against a customer-defined ontology:

```bash
opf train /path/to/train.jsonl \
  --validation-dataset /path/to/validation.jsonl \
  --label-space-json /path/to/custom_label_space.json \
  --output-dir /path/to/finetuned_checkpoint
```

Example `custom_label_space.json`:

```json
{
  "category_version": "custom_v1",
  "span_class_names": ["O", "custom_account_id", "custom_secret"]
}
```

Notes:
- `span_class_names` is preferred.
- `O` must be present as the first entry.

## Output Artifacts

`--output-dir` writes:
- `config.json`
- `model.safetensors`
- `finetune_summary.json`
- `USAGE.txt`

## Reproducible Demo Harness Scripts

Canonical harness scripts live in `examples/scripts/finetuning/`:

1. `finetune_secret_demo.sh`
   - The demo dataset contains strings redacted as <ACCOUNT_NUMBER> by the baseline model, but with ground truths labeled as <SECRET> instead. It illustrates how the model can be retrained to adapt to  to the category policy change.
2. `finetune_custom_label_demo.sh`
   - The demo defines a new label space consisting only of the background class ("O") and a single, newly defined "custom_secret" category. It illustrates how the model can be adapted to recognize this new category instead of its original categories.

These demos intentionally use tiny toy splits (1 example per split) and higher
default epoch counts so the before/after behavior is easy to inspect.

Each harness accepts:
- `--checkpoint` (required)
- optional `--workdir` (logs/metrics/artifacts)
- optional `--output-checkpoint-dir` (exact finetuned checkpoint path)

If `--output-checkpoint-dir` is omitted, checkpoint output defaults to:

`<workdir>/finetuned_checkpoint`
