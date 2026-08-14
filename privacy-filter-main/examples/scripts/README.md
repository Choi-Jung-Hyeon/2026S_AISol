# Finetuning Harness Scripts

Canonical script entrypoints (in `examples/scripts/finetuning/`):

- `finetune_secret_demo.sh`
  - Baseline vs finetuned `secret` behavior on fixed held-out data.
- `finetune_custom_label_demo.sh`
  - Single custom-label-space adaptation (`custom_secret`).

All demos use tiny toy splits (1 example per split) and higher default epoch
counts to make qualitative before/after behavior obvious.

Common options:

- `--checkpoint <BASE_CHECKPOINT_DIR>` (required)
- `--workdir <ARTIFACT_DIR>` (optional, default is a timestamped `/tmp/...` path)
- `--output-checkpoint-dir <CHECKPOINT_DIR>` (optional, defaults to `<workdir>/finetuned_checkpoint`)
- `--preview-examples <N>` (optional)
