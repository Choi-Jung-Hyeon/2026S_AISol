"""정본·프로브셋을 bfloat16 으로 1회 추론해 로짓 [T,33] 과 토큰 오프셋을 캐시한다.

이후 argmax 와 Viterbi 두 디코딩은 전부 이 캐시에서 낸다. 재추론하지 않는다.
캐시는 리포 밖(~/.opf/cache)에 둔다 — 용량이 크고 git 이 추적하면 안 된다.

    python3 cache_logits.py --which gold
    python3 cache_logits.py --which probe

로짓은 float16 으로 저장한다. 추론은 bfloat16 으로 하되 저장 시 float16 으로
내리면 용량이 절반이고, argmax·Viterbi 판정에 필요한 유효숫자는 보존된다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import opf_local

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "data", "ss_pii_testset_ko_v1.json")
PROBE = os.path.join(HERE, "data", "probe_partial_utterance_corp.jsonl")
CACHE_DIR = os.path.expanduser("~/.opf/cache")


def load_gold_docs(path):
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    return [(d["id"], d["text"]) for d in docs]


def load_probe_docs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append((r["info"]["id"], r["text"]))
    return out


def main():
    ap = argparse.ArgumentParser(prog="cache_logits.py")
    ap.add_argument("--which", required=True, choices=["gold", "probe"])
    ap.add_argument("--ckpt", default=opf_local.DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=CACHE_DIR)
    args = ap.parse_args()

    import torch

    if args.which == "gold":
        if not os.path.isfile(GOLD):
            sys.exit("정본 JSON 이 없습니다: %s" % GOLD)
        items = load_gold_docs(GOLD)
    else:
        if not os.path.isfile(PROBE):
            sys.exit("프로브셋이 없습니다: %s" % PROBE)
        items = load_probe_docs(PROBE)

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, "logits_%s.npz" % args.which)
    meta_path = os.path.join(args.out_dir, "meta_%s.jsonl" % args.which)

    tokenizer = opf_local.load_tokenizer(args.ckpt)
    backend = opf_local.LocalBackend(args.ckpt, None, torch.bfloat16)
    print("device %s / dtype bfloat16 / 문서 %d건" % (backend.device, len(items)))
    print("캐시 경로: %s" % args.out_dir)

    store, t0, n_tok = {}, time.time(), 0
    with open(meta_path, "w", encoding="utf-8") as w:
        for k, (did, text) in enumerate(items):
            enc = tokenizer.encode(text)
            lg = backend.logits(enc.ids)
            if lg.shape[0] != len(enc.offsets):
                sys.exit("토큰 수 불일치 %s: %d != %d"
                         % (did, lg.shape[0], len(enc.offsets)))
            store[did] = lg.astype(np.float16)
            n_tok += lg.shape[0]
            w.write(json.dumps({
                "doc_id": did,
                "text": text,
                "offsets": [[int(a), int(b)] for a, b in enc.offsets],
            }, ensure_ascii=False) + "\n")
            if (k + 1) % 500 == 0:
                print("  ... %d/%d (%.1f분)"
                      % (k + 1, len(items), (time.time() - t0) / 60), flush=True)

    np.savez(npz_path, **store)
    el = time.time() - t0
    size = os.path.getsize(npz_path) / 1e6
    print("완료 %d건 / 토큰 %d / %.1f분 (건당 %.3f초)"
          % (len(items), n_tok, el / 60, el / len(items)))
    print("  로짓 캐시 : %s  (%.1f MB)" % (npz_path, size))
    print("  메타      : %s  (%.1f MB)"
          % (meta_path, os.path.getsize(meta_path) / 1e6))


if __name__ == "__main__":
    main()
