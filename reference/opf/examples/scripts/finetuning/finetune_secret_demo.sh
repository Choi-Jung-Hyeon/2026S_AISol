#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/examples/data/finetuning_secret_demo"

CHECKPOINT=""
DEVICE="cpu"
EPOCHS="40"
BATCH_SIZE="1"
LEARNING_RATE="2e-4"
MIN_DELTA="0.50"
MIN_POST_F1="0.90"
PREVIEW_EXAMPLES="1"
REUSE_ARTIFACTS="0"
WORKDIR=""
FINETUNE_DIR=""

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --checkpoint /path/to/checkpoint_dir [options]

Options:
  --checkpoint PATH   Base checkpoint to finetune (required)
  --device DEVICE     Device for train/eval (default: cpu)
  --epochs N          Finetune epochs (default: 40)
  --batch-size N      Finetune batch size (default: 1)
  --learning-rate LR  Finetune learning rate (default: 2e-4)
  --workdir PATH      Artifact/log directory (default: /tmp/opf_finetune_secret_demo_<ts>)
  --output-checkpoint-dir PATH  Finetuned checkpoint output dir (default: <workdir>/finetuned_checkpoint)
  --min-delta F       Required improvement in secret span F1 (default: 0.50)
  --min-post-f1 F     Required final secret span F1 (default: 0.90)
  --preview-examples N  Deprecated no-op; retained for backward compatibility
  --reuse-artifacts   Skip baseline/train/post runs and reuse existing files in --workdir

This script enforces reproducible best-practice checks:
  - fixed train/validation/test splits
  - baseline vs post-finetune evaluation on untouched test set
  - explicit quality gates on test metrics
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
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --output-checkpoint-dir)
      FINETUNE_DIR="$2"
      shift 2
      ;;
    --min-delta)
      MIN_DELTA="$2"
      shift 2
      ;;
    --min-post-f1)
      MIN_POST_F1="$2"
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

if [[ -z "${WORKDIR}" ]]; then
  WORKDIR="/tmp/opf_finetune_secret_demo_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${WORKDIR}"
if [[ -z "${FINETUNE_DIR}" ]]; then
  FINETUNE_DIR="${WORKDIR}/finetuned_checkpoint"
fi

BASELINE_METRICS="${WORKDIR}/baseline_metrics.json"
POST_METRICS="${WORKDIR}/post_metrics.json"

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

echo "Finetuning Privacy Filter: Demo 1."
echo "The demo dataset contains strings redacted as <ACCOUNT_NUMBER> by the baseline model, but with ground truths labeled as <SECRET> instead. It illustrates how the model can be retrained to adapt to  to the category policy change."
echo "Training data: ${DATA_DIR}/train.jsonl"
echo "Validation data: ${DATA_DIR}/validation.jsonl"
echo "Test data: ${DATA_DIR}/test.jsonl"
echo "Output path: ${WORKDIR}"

if [[ "${REUSE_ARTIFACTS}" == "1" ]]; then
  echo "Reusing existing artifacts in ${WORKDIR}."
  missing=0
  for required in \
    "${BASELINE_METRICS}" \
    "${POST_METRICS}" \
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
  if ! python -m opf eval "${DATA_DIR}/test.jsonl" \
      --checkpoint "${CHECKPOINT}" \
      --device "${DEVICE}" \
      --eval-mode typed \
      --n-ctx 128 \
      --window-batch-size 1 \
      --preprocess-workers 1 \
      --metrics-out "${BASELINE_METRICS}" \
      > "${WORKDIR}/baseline_eval.log" 2>&1; then
    echo "Baseline eval failed. See ${WORKDIR}/baseline_eval.log" >&2
    exit 2
  fi

  echo "Finetuning progress:"
  if ! python -m opf train "${DATA_DIR}/train.jsonl" \
      --checkpoint "${CHECKPOINT}" \
      --validation-dataset "${DATA_DIR}/validation.jsonl" \
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
      2>&1 | tee "${WORKDIR}/train.log" | python -c '
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
    echo "Finetuning failed. See ${WORKDIR}/train.log" >&2
    exit 2
  fi

  if ! python -m opf eval "${DATA_DIR}/test.jsonl" \
      --checkpoint "${FINETUNE_DIR}" \
      --device "${DEVICE}" \
      --eval-mode typed \
      --n-ctx 128 \
      --window-batch-size 1 \
      --preprocess-workers 1 \
      --metrics-out "${POST_METRICS}" \
      > "${WORKDIR}/post_eval.log" 2>&1; then
    echo "Post-finetune eval failed. See ${WORKDIR}/post_eval.log" >&2
    exit 2
  fi
fi

python - <<'PY' "${BASELINE_METRICS}" "${POST_METRICS}" "${MIN_DELTA}" "${MIN_POST_F1}"
import json
import re
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
post_path = Path(sys.argv[2])
min_delta = float(sys.argv[3])
min_post_f1 = float(sys.argv[4])

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

    # Some eval metrics are omitted when the denominator is zero
    # (for example precision/f1 when there are no positive predictions).
    # For baseline-vs-finetune comparisons we treat missing as zero, and
    # derive f-scores from precision/recall when available.
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

baseline = load_metrics(baseline_path)
post = load_metrics(post_path)

baseline_secret_f1 = metric(baseline, "by_class.secret.span.f1")
post_secret_f1 = metric(post, "by_class.secret.span.f1")
delta = post_secret_f1 - baseline_secret_f1

if post_secret_f1 < min_post_f1:
    raise SystemExit(
        f"FAIL: post-finetune secret span F1 {post_secret_f1:.4f} < required {min_post_f1:.4f}"
    )
if delta < min_delta:
    raise SystemExit(
        f"FAIL: secret span F1 delta {delta:.4f} < required {min_delta:.4f}"
    )

span_recall_category_pattern = re.compile(r"^by_class\.([^.]+)\.span\.recall$")
boundary_tag_pattern = re.compile(r"^by_class\.(B|I|E|S)-([^.]+)\.(precision|recall|f1)$")

all_keys = set(baseline) | set(post)
categories: set[str] = set()
token_tags_by_category: dict[str, set[str]] = {}

for key in all_keys:
    span_match = span_recall_category_pattern.match(key)
    if span_match:
        categories.add(span_match.group(1))
    token_match = boundary_tag_pattern.match(key)
    if token_match:
        tag = token_match.group(1)
        category = token_match.group(2)
        token_tags_by_category.setdefault(category, set()).add(tag)

def token_metric_for_category(metrics: dict[str, float], category: str, suffix: str) -> float:
    tags = sorted(token_tags_by_category.get(category, set()))
    if tags:
        values = [metric(metrics, f"by_class.{tag}-{category}.{suffix}") for tag in tags]
        return sum(values) / float(len(values))
    return metric(metrics, f"by_class.{category}.{suffix}")

def span_metric_for_category(metrics: dict[str, float], category: str, suffix: str) -> float:
    return metric(metrics, f"by_class.{category}.span.{suffix}")

def fmt_transition(before: float, after: float) -> str:
    return f"{before:.4f} -> {after:.4f}"

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
        fmt_transition(metric(baseline, "detection.f1"), metric(post, "detection.f1")),
        fmt_transition(metric(baseline, "detection.span.f1"), metric(post, "detection.span.f1")),
        fmt_transition(metric(baseline, "detection.recall"), metric(post, "detection.recall")),
        fmt_transition(metric(baseline, "detection.span.recall"), metric(post, "detection.span.recall")),
        fmt_transition(metric(baseline, "detection.precision"), metric(post, "detection.precision")),
        fmt_transition(metric(baseline, "detection.span.precision"), metric(post, "detection.span.precision")),
    ]
)
for category in sorted(categories):
    rows.append(
        [
            category,
            fmt_transition(
                token_metric_for_category(baseline, category, "f1"),
                token_metric_for_category(post, category, "f1"),
            ),
            fmt_transition(
                span_metric_for_category(baseline, category, "f1"),
                span_metric_for_category(post, category, "f1"),
            ),
            fmt_transition(
                token_metric_for_category(baseline, category, "recall"),
                token_metric_for_category(post, category, "recall"),
            ),
            fmt_transition(
                span_metric_for_category(baseline, category, "recall"),
                span_metric_for_category(post, category, "recall"),
            ),
            fmt_transition(
                token_metric_for_category(baseline, category, "precision"),
                token_metric_for_category(post, category, "precision"),
            ),
            fmt_transition(
                span_metric_for_category(baseline, category, "precision"),
                span_metric_for_category(post, category, "precision"),
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
