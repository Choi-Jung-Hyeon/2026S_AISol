# finetuning_custom_label_demo

Synthetic dataset for custom-label finetuning demos.

## Purpose

This dataset demonstrates adapting a base checkpoint to a customer label (`custom_secret`) and validating on a held-out split.

## Split sizes

- `train.jsonl`: 1 example
- `validation.jsonl`: 1 example
- `test.jsonl`: 1 example

## Notes

- train/validation/test marker codes are disjoint to prevent leakage
- labels follow the same schema expected by `opf train` and `opf eval`
- `label_space.json` defines the custom ontology (`category_version=custom_v1`)
