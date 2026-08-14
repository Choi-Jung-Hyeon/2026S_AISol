#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/examples/data/finetuning_custom_label_demo"

CHECKPOINT=""
DEVICE="cpu"
EPOCHS="40"
BATCH_SIZE="1"
LEARNING_RATE="2e-4"
MIN_POST_TYPED_F1="0.90"
MIN_UNTYPED_DELTA="0.00"
PREVIEW_EXAMPLES="1"
REUSE_ARTIFACTS="0"
WORKDIR=""
LABEL_SPACE_JSON="${DATA_DIR}/label_space.json"
FINETUNE_DIR=""

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --checkpoint /path/to/checkpoint_dir [options]

Options:
  --checkpoint PATH        Base checkpoint to finetune (required)
  --device DEVICE          Device for train/eval (default: cpu)
  --epochs N               Finetune epochs (default: 40)
  --batch-size N           Finetune batch size (default: 1)
  --learning-rate LR       Finetune learning rate (default: 2e-4)
  --label-space-json PATH  Custom label-space json (default: examples/data/finetuning_custom_label_demo/label_space.json)
  --workdir PATH           Artifact/log directory (default: /tmp/opf_finetune_custom_label_demo_<ts>)
  --output-checkpoint-dir PATH  Finetuned checkpoint output dir (default: <workdir>/finetuned_checkpoint)
  --min-post-typed-f1 F    Required final typed F1 for the custom target label (default: 0.90)
  --min-untyped-delta F    Required improvement in untyped detection.span.f1 (default: 0.00)
  --preview-examples N     Deprecated no-op; retained for backward compatibility
  --reuse-artifacts        Skip baseline/train/post runs and reuse existing files in --workdir

This script provides reproducible evidence that custom label-space finetuning works:
  - fixed train/validation/test splits
  - baseline typed eval behavior when labels are unknown to the base checkpoint
  - baseline vs post-finetune untyped held-out metrics
  - post-finetune typed held-out metrics for the custom label
  - before/after redaction previews and predicted spans
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --label-space-json)
      LABEL_SPACE_JSON="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --output-checkpoint-dir)
      FINETUNE_DIR="$2"
      shift 2
      ;;
    --min-post-typed-f1)
      MIN_POST_TYPED_F1="$2"
      shift 2
      ;;
    --min-untyped-delta)
      MIN_UNTYPED_DELTA="$2"
      shift 2
      ;;
    --preview-examples)
      PREVIEW_EXAMPLES="$2"
      shift 2
      ;;
    --reuse-artifacts)
      REUSE_ARTIFACTS="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "--checkpoint is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${LABEL_SPACE_JSON}" ]]; then
  echo "Label-space JSON not found: ${LABEL_SPACE_JSON}" >&2
  exit 2
fi

if [[ -z "${WORKDIR}" ]]; then
  WORKDIR="/tmp/opf_finetune_custom_label_demo_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${WORKDIR}"
if [[ -z "${FINETUNE_DIR}" ]]; then
  FINETUNE_DIR="${WORKDIR}/finetuned_checkpoint"
fi

BASELINE_TYPED_STATUS="${WORKDIR}/baseline_typed_status.txt"
BASELINE_TYPED_LOG="${WORKDIR}/baseline_typed_eval.log"
BASELINE_TYPED_METRICS="${WORKDIR}/baseline_typed_metrics.json"
BASELINE_TYPED_PRED="${WORKDIR}/baseline_typed_predictions.jsonl"

BASELINE_UNTYPED_METRICS="${WORKDIR}/baseline_untyped_metrics.json"
BASELINE_UNTYPED_PRED="${WORKDIR}/baseline_untyped_predictions.jsonl"
BASELINE_UNTYPED_LOG="${WORKDIR}/baseline_untyped_eval.log"

TRAIN_LOG="${WORKDIR}/train.log"

POST_TYPED_METRICS="${WORKDIR}/post_typed_metrics.json"
POST_TYPED_PRED="${WORKDIR}/post_typed_predictions.jsonl"
POST_TYPED_LOG="${WORKDIR}/post_typed_eval.log"
POST_UNTYPED_METRICS="${WORKDIR}/post_untyped_metrics.json"
POST_UNTYPED_PRED="${WORKDIR}/post_untyped_predictions.jsonl"
POST_UNTYPED_LOG="${WORKDIR}/post_untyped_eval.log"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export KMP_USE_SHM=0
export DD_TRACE_ENABLED=false
export DD_PROFILING_ENABLED=false

cd "${REPO_ROOT}"

for split in train validation test; do
  if [[ ! -f "${DATA_DIR}/${split}.jsonl" ]]; then
    echo "missing dataset split file: ${DATA_DIR}/${split}.jsonl" >&2
    exit 2
  fi
done

echo "Finetuning Privacy Filter: Demo 2."
echo "The demo defines a new label space consisting only of the background class (\"O\") and a single, newly defined \"custom_secret\" category. It illustrates how the model can be adapted to recognize this new category instead of its original categories."
echo "Training data: ${DATA_DIR}/train.jsonl"
echo "Validation data: ${DATA_DIR}/validation.jsonl"
echo "Test data: ${DATA_DIR}/test.jsonl"
echo "Output path: ${WORKDIR}"

if [[ "${REUSE_ARTIFACTS}" == "1" ]]; then
  echo "Reusing existing artifacts in ${WORKDIR}."
  missing=0
  for required in \
    "${BASELINE_TYPED_STATUS}" \
    "${BASELINE_UNTYPED_METRICS}" \
    "${BASELINE_UNTYPED_PRED}" \
    "${POST_TYPED_METRICS}" \
    "${POST_TYPED_PRED}" \
    "${POST_UNTYPED_METRICS}" \
    "${POST_UNTYPED_PRED}" \
    "${FINETUNE_DIR}/config.json" \
    "${FINETUNE_DIR}/model.safetensors"; do
    if [[ ! -f "${required}" ]]; then
      echo "Missing required artifact: ${required}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    echo "Cannot reuse artifacts because required files are missing." >&2
    exit 2
  fi
else
  if python -m opf eval "${DATA_DIR}/test.jsonl" \
      --checkpoint "${CHECKPOINT}" \
      --device "${DEVICE}" \
      --eval-mode typed \
      --n-ctx 128 \
      --window-batch-size 1 \
      --preprocess-workers 0 \
      --metrics-out "${BASELINE_TYPED_METRICS}" \
      --predictions-out "${BASELINE_TYPED_PRED}" \
      --prediction-write-workers 0 \
      > "${BASELINE_TYPED_LOG}" 2>&1; then
    echo "success" > "${BASELINE_TYPED_STATUS}"
  else
    echo "failed" > "${BASELINE_TYPED_STATUS}"
  fi

  if ! python -m opf eval "${DATA_DIR}/test.jsonl" \
      --checkpoint "${CHECKPOINT}" \
      --device "${DEVICE}" \
      --eval-mode untyped \
      --n-ctx 128 \
      --window-batch-size 1 \
      --preprocess-workers 0 \
      --metrics-out "${BASELINE_UNTYPED_METRICS}" \
      --predictions-out "${BASELINE_UNTYPED_PRED}" \
      --prediction-write-workers 0 \
      > "${BASELINE_UNTYPED_LOG}" 2>&1; then
    echo "Baseline untyped eval failed. See ${BASELINE_UNTYPED_LOG}" >&2
    exit 2
  fi

  echo "Finetuning progress:"
  if ! python -m opf train "${DATA_DIR}/train.jsonl" \
      --checkpoint "${CHECKPOINT}" \
      --validation-dataset "${DATA_DIR}/validation.jsonl" \
      --label-space-json "${LABEL_SPACE_JSON}" \
      --output-dir "${FINETUNE_DIR}" \
      --overwrite-output \
      --device "${DEVICE}" \
      --n-ctx 128 \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum-steps 1 \
      --learning-rate "${LEARNING_RATE}" \
      --weight-decay 0.0 \
      --max-grad-norm 1.0 \
      --shuffle-seed 1337 \
      2>&1 | tee "${TRAIN_LOG}" | python -c '
import re
import sys

last = -1

def render_progress(pct: int) -> None:
    width = 30
    filled = int((pct * width) / 100)
    bar = ("#" * filled) + ("-" * (width - filled))
    sys.stdout.write(f"\r[{bar}] {pct:3d}%")
    sys.stdout.flush()

for line in sys.stdin:
    m = re.search(r"epoch=(\d+)/(\d+)", line)
    if m is None:
        m = re.search(r"epoch (\d+)/(\d+):", line)
    if m is None:
        continue
    current = int(m.group(1))
    total = int(m.group(2))
    if total <= 0:
        continue
    pct = int((current * 100) / total)
    if pct > last:
        render_progress(pct)
        last = pct
if last >= 0 and last < 100:
    render_progress(100)
if last >= 0:
    sys.stdout.write("\n")
    sys.stdout.flush()
'; then
    echo "Finetuning failed. See ${TRAIN_LOG}" >&2
    exit 2
  fi

  if ! python -m opf eval "${DATA_DIR}/test.jsonl" \
      --checkpoint "${FINETUNE_DIR}" \
      --device "${DEVICE}" \
      --eval-mode typed \
      --n-ctx 128 \
      --window-batch-size 1 \
      --preprocess-workers 0 \
      --metrics-out "${POST_TYPED_METRICS}" \
      --predictions-out "${POST_TYPED_PRED}" \
      --prediction-write-workers 0 \
      > "${POST_TYPED_LOG}" 2>&1; then
    echo "Post-finetune typed eval failed. See ${POST_TYPED_LOG}" >&2
    exit 2
  fi

  if ! python -m opf eval "${DATA_DIR}/test.jsonl" \
      --checkpoint "${FINETUNE_DIR}" \
      --device "${DEVICE}" \
      --eval-mode untyped \
      --n-ctx 128 \
      --window-batch-size 1 \
      --preprocess-workers 0 \
      --metrics-out "${POST_UNTYPED_METRICS}" \
      --predictions-out "${POST_UNTYPED_PRED}" \
      --prediction-write-workers 0 \
      > "${POST_UNTYPED_LOG}" 2>&1; then
    echo "Post-finetune untyped eval failed. See ${POST_UNTYPED_LOG}" >&2
    exit 2
  fi
fi
python - <<'PY' \
  "${BASELINE_TYPED_STATUS}" \
  "${BASELINE_TYPED_METRICS}" \
  "${BASELINE_UNTYPED_METRICS}" \
  "${POST_TYPED_METRICS}" \
  "${POST_UNTYPED_METRICS}" \
  "${LABEL_SPACE_JSON}" \
  "${FINETUNE_DIR}" \
  "${MIN_POST_TYPED_F1}" \
  "${MIN_UNTYPED_DELTA}"
import json
import re
import sys
from pathlib import Path

baseline_typed_status_path = Path(sys.argv[1])
baseline_typed_metrics_path = Path(sys.argv[2])
baseline_untyped_metrics_path = Path(sys.argv[3])
post_typed_metrics_path = Path(sys.argv[4])
post_untyped_metrics_path = Path(sys.argv[5])
label_space_path = Path(sys.argv[6])
finetuned_checkpoint = Path(sys.argv[7])
min_post_typed_f1 = float(sys.argv[8])
min_untyped_delta = float(sys.argv[9])


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"invalid metrics payload in {path}")
    return metrics


def metric(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key)
    if value is not None:
        return float(value)
    if key.endswith(".f1"):
        prefix = key[:-3]
        precision = float(metrics.get(f"{prefix}.precision", 0.0))
        recall = float(metrics.get(f"{prefix}.recall", 0.0))
        denom = precision + recall
        return 0.0 if denom <= 0.0 else (2.0 * precision * recall) / denom
    if key.endswith(".f2"):
        prefix = key[:-3]
        precision = float(metrics.get(f"{prefix}.precision", 0.0))
        recall = float(metrics.get(f"{prefix}.recall", 0.0))
        beta2 = 4.0
        denom = (beta2 * precision) + recall
        return 0.0 if denom <= 0.0 else ((1.0 + beta2) * precision * recall) / denom
    return 0.0


def token_metric_for_category(metrics: dict[str, float], category: str, suffix: str) -> float:
    tags = []
    for tag in ("B", "I", "E", "S"):
        key = f"by_class.{tag}-{category}.{suffix}"
        if key in metrics:
            tags.append(tag)
    if tags:
        values = [metric(metrics, f"by_class.{tag}-{category}.{suffix}") for tag in tags]
        return sum(values) / float(len(values))
    return metric(metrics, f"by_class.{category}.{suffix}")


def span_metric_for_category(metrics: dict[str, float], category: str, suffix: str) -> float:
    return metric(metrics, f"by_class.{category}.span.{suffix}")


def fmt_transition(before: float, after: float) -> str:
    return f"{before:.4f} -> {after:.4f}"


label_space_payload = json.loads(label_space_path.read_text(encoding="utf-8"))
span_class_names = label_space_payload.get("span_class_names")
if not isinstance(span_class_names, list) or not span_class_names:
    raise SystemExit(f"invalid or missing span_class_names in {label_space_path}")
if not isinstance(span_class_names[0], str) or span_class_names[0] != "O":
    raise SystemExit("label-space first span_class_names entry must be 'O'")
target_labels = [name for name in span_class_names if isinstance(name, str) and name != "O"]
if not target_labels:
    raise SystemExit("label-space must include at least one non-background label")
target_label = target_labels[0]
expected_category_version = str(label_space_payload.get("category_version", "custom"))

finetuned_config = json.loads((finetuned_checkpoint / "config.json").read_text(encoding="utf-8"))
resolved_category_version = str(finetuned_config.get("category_version", ""))
resolved_span_class_names = finetuned_config.get("span_class_names")
if resolved_category_version != expected_category_version:
    raise SystemExit(
        "FAIL: finetuned checkpoint category_version mismatch: "
        f"expected {expected_category_version!r}, got {resolved_category_version!r}"
    )
if not isinstance(resolved_span_class_names, list) or target_label not in resolved_span_class_names:
    raise SystemExit(
        "FAIL: finetuned checkpoint span_class_names missing custom target label "
        f"{target_label!r}"
    )

baseline_untyped = load_metrics(baseline_untyped_metrics_path)
post_typed = load_metrics(post_typed_metrics_path)
post_untyped = load_metrics(post_untyped_metrics_path)

baseline_typed_succeeded = (
    baseline_typed_status_path.read_text(encoding="utf-8").strip().lower() == "success"
)
baseline_typed: dict[str, float] | None = None
if baseline_typed_succeeded and baseline_typed_metrics_path.exists():
    baseline_typed = load_metrics(baseline_typed_metrics_path)

post_typed_f1 = metric(post_typed, f"by_class.{target_label}.span.f1")
if post_typed_f1 < min_post_typed_f1:
    raise SystemExit(
        f"FAIL: post-finetune typed custom-label F1 {post_typed_f1:.4f} < required {min_post_typed_f1:.4f}"
    )

untyped_delta = metric(post_untyped, "detection.span.f1") - metric(
    baseline_untyped, "detection.span.f1"
)
if untyped_delta < min_untyped_delta:
    raise SystemExit(
        f"FAIL: untyped detection.span.f1 delta {untyped_delta:.4f} < required {min_untyped_delta:.4f}"
    )

baseline_overall = baseline_typed if baseline_typed is not None else baseline_untyped

span_recall_category_pattern = re.compile(r"^by_class\.([^.]+)\.span\.recall$")
categories = set(target_labels)
for key in post_typed:
    match = span_recall_category_pattern.match(key)
    if match:
        categories.add(match.group(1))

headers = [
    "Category",
    "F1 (tokens)",
    "F1 (spans)",
    "Recall (tokens)",
    "Recall (spans)",
    "Precision (tokens)",
    "Precision (spans)",
]

rows: list[list[str]] = []
rows.append(
    [
        "any",
        fmt_transition(metric(baseline_overall, "detection.f1"), metric(post_typed, "detection.f1")),
        fmt_transition(metric(baseline_overall, "detection.span.f1"), metric(post_typed, "detection.span.f1")),
        fmt_transition(metric(baseline_overall, "detection.recall"), metric(post_typed, "detection.recall")),
        fmt_transition(metric(baseline_overall, "detection.span.recall"), metric(post_typed, "detection.span.recall")),
        fmt_transition(metric(baseline_overall, "detection.precision"), metric(post_typed, "detection.precision")),
        fmt_transition(metric(baseline_overall, "detection.span.precision"), metric(post_typed, "detection.span.precision")),
    ]
)

for category in sorted(categories):
    baseline_category = baseline_typed if baseline_typed is not None else {}
    rows.append(
        [
            category,
            fmt_transition(
                token_metric_for_category(baseline_category, category, "f1"),
                token_metric_for_category(post_typed, category, "f1"),
            ),
            fmt_transition(
                span_metric_for_category(baseline_category, category, "f1"),
                span_metric_for_category(post_typed, category, "f1"),
            ),
            fmt_transition(
                token_metric_for_category(baseline_category, category, "recall"),
                token_metric_for_category(post_typed, category, "recall"),
            ),
            fmt_transition(
                span_metric_for_category(baseline_category, category, "recall"),
                span_metric_for_category(post_typed, category, "recall"),
            ),
            fmt_transition(
                token_metric_for_category(baseline_category, category, "precision"),
                token_metric_for_category(post_typed, category, "precision"),
            ),
            fmt_transition(
                span_metric_for_category(baseline_category, category, "precision"),
                span_metric_for_category(post_typed, category, "precision"),
            ),
        ]
    )

widths = [len(header) for header in headers]
for row in rows:
    for idx, value in enumerate(row):
        if len(value) > widths[idx]:
            widths[idx] = len(value)


def border() -> str:
    return "+" + "+".join("-" * (width + 2) for width in widths) + "+"


def render_row(cells: list[str]) -> str:
    padded = [cells[idx].ljust(widths[idx]) for idx in range(len(cells))]
    return "| " + " | ".join(padded) + " |"


print("Finetuning run complete.")
print("")
print(border())
print(render_row(headers))
print(border())
for row in rows:
    print(render_row(row))
print(border())
PY
