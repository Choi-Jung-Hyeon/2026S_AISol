# OPF Output Schemas

## 1. `opf` / `opf redact` JSON Output

Printed to stdout once per input example.

```json
{
  "schema_version": 1,
  "summary": {
    "output_mode": "typed",
    "span_count": 3,
    "by_label": {
      "private_person": 1,
      "private_date": 2
    },
    "decoded_mismatch": false
  },
  "text": "Alice was born on 1990-01-02.",
  "detected_spans": [
    {
      "label": "private_person",
      "start": 0,
      "end": 5,
      "text": "Alice",
      "placeholder": "<PRIVATE_PERSON>"
    }
  ],
  "redacted_text": "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
}
```

Notes:

- In `--output-mode redacted`, every `detected_spans[*].label` becomes `redacted`.
- `warning` is present only when tokenizer decode does not exactly round-trip the input text.

## 2. `opf eval` Predictions Output (`--predictions-out`)

Written as JSONL when requested.

```json
{
  "example_id": "stable-id",
  "text": "Alice was born on 1990-01-02.",
  "predicted_spans": {
    "private_person: Alice": [[0, 5]]
  }
}
```

Optional field:

- `token_logprobs_topk`: included only when `--predictions-token-logprobs-topk > 0`

Notes:

- This file is literal JSONL: one compact JSON object per line.

## Stability Notes

- `typed` and `untyped` are the evaluation terms.
- `typed` and `redacted` are the prediction-output terms.
- Additive fields may appear over time, but existing keys should remain stable unless `schema_version` changes for API/CLI JSON payloads.
