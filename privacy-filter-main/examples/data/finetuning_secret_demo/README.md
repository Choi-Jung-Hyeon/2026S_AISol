# finetuning_secret_demo

Synthetic dataset for local finetuning demos.

## Purpose

This dataset is designed for a reproducible baseline-vs-finetune demonstration with `opf train`:

- train split: optimization only
- validation split: model selection during finetune
- test split: untouched holdout for before/after comparison

## Schema

Each JSONL record uses the same schema as `opf eval` and `opf train`:

- `text`: input string
- `label`: list of entities with `category`, `start`, `end`
- `info`: metadata (`id`, `split`, and marker code)

The target category is `secret`.

## Split sizes

- `train.jsonl`: 1 example
- `validation.jsonl`: 1 example
- `test.jsonl`: 1 example

## Validation checks

- secret marker codes are disjoint across train/validation/test to prevent leakage
- holdout test is never used for finetuning
- offsets are character-based and exact span boundaries
- dataset content is deterministic
