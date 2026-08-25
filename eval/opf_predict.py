"""정본 전량에 로컬 OPF 추론을 돌려 예측 스팬 JSONL 을 만든다.

산출물은 기존 집계 스크립트(standalone_metrics.py 등)가 그대로 먹는
하니스 predictions 포맷이라, run_all.py --pred 로 이어 붙일 수 있다.

    python3 opf_predict.py --out results/opf_local_preds.jsonl
    python3 opf_predict.py --limit 300 --out results/opf_smoke300.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import opf_local

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "data", "ss_pii_testset_ko_v1.json")


def main():
    ap = argparse.ArgumentParser(prog="opf_predict.py")
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞 N건만 (0=전량)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=opf_local.DEFAULT_CKPT)
    args = ap.parse_args()

    if not os.path.isfile(args.gold):
        sys.exit("정본 JSON 이 없습니다: %s" % args.gold)

    with open(args.gold, encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    if args.limit:
        docs = docs[: args.limit]

    tokenizer, backend, id2label = opf_local.build(
        "local", args.ckpt, args.device)
    print("device %s / 문서 %d건" % (getattr(backend, "device", "-"), len(docs)))

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    n_span = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for k, d in enumerate(docs):
            enc, tags, _ = opf_local.tag_tokens(
                d["text"], tokenizer, backend, id2label)
            spans = opf_local.group_spans(tags, enc.offsets)
            n_span += len(spans)
            w.write(json.dumps({
                "text": d["text"],
                "predictions": [
                    {"start": s["start"], "end": s["end"], "label": s["label"]}
                    for s in spans
                ],
            }, ensure_ascii=False) + "\n")
            if (k + 1) % 500 == 0:
                print("  ... %d/%d (%.1f분)"
                      % (k + 1, len(docs), (time.time() - t0) / 60), flush=True)

    el = time.time() - t0
    print("완료 %d건 / 예측스팬 %d개 / %.1f분 (건당 %.3f초)"
          % (len(docs), n_span, el / 60, el / len(docs)))
    print("출력: %s" % args.out)


if __name__ == "__main__":
    main()
