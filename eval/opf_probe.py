"""OPF 로컬 추론 검증기 — 토큰·오프셋·예측태그를 전수 덤프한다.

주소가 한 글자만 마스킹되는 원인이 토크나이저 경계 때문인지,
태그 시퀀스가 중간에 O 로 끊기기 때문인지를 눈으로 가리기 위한 도구다.

    python3 opf_probe.py --text "My name is Alice Marie Smith"
    python3 opf_probe.py --doc-ids SS-KO-02337 SS-KO-00693
    python3 opf_probe.py --bench 20        # 속도 실측
    python3 opf_probe.py --repeat 5 --text "..."   # 재현성
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


def load_gold(path=GOLD):
    if not os.path.isfile(path):
        sys.exit("정본 JSON 이 없습니다: %s" % path)
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in json.load(f)["documents"]}


def flat(s):
    return s.replace("\n", "\\n").replace("\t", "\\t")


def dump_tokens(text, enc, tags, gold_spans=None):
    """토큰 단위 (토큰문자열, 오프셋, 예측태그) 를 전부 찍는다."""
    cover = set()
    if gold_spans:
        for g in gold_spans:
            cover.update(range(g["start"], g["end"]))

    print("  %-5s %-22s %-13s %-22s %-4s %s"
          % ("#", "token", "offset", "pred_tag", "gold", "원문조각"))
    print("  " + "-" * 96)
    for i, (tok, (s, e), tag) in enumerate(zip(enc.tokens, enc.offsets, tags)):
        in_gold = "■" if any(p in cover for p in range(s, e)) else ""
        mark = "" if tag == "O" else "◀"
        print("  %-5d %-22s %-13s %-13s %-4s %-4s %r"
              % (i, flat(tok)[:22], "(%d,%d)" % (s, e), tag, mark, in_gold,
                 flat(text[s:e])[:22]))


def show_doc(doc, tokenizer, backend, id2label, note=""):
    text = doc["text"]
    enc, tags, _ = opf_local.tag_tokens(text, tokenizer, backend, id2label)
    print("=" * 100)
    print("%s  %s" % (doc["id"], note))
    print("  문자 %d / 토큰 %d / 정답스팬 %d"
          % (len(text), len(enc.ids), len(doc["spans"])))
    print("=" * 100)
    dump_tokens(text, enc, tags, doc["spans"])

    spans = opf_local.group_spans(tags, enc.offsets)
    masked = opf_local.mask_text(text, spans)
    print()
    print("  [예측 스팬 %d개]" % len(spans))
    for sp in spans:
        print("     [%4d,%4d) %-16s %r"
              % (sp["start"], sp["end"], sp["label"],
                 flat(text[sp["start"]:sp["end"]])[:44]))
    print()
    print("  [정답 스팬 %d개 — 예측이 덮은 구간]" % len(doc["spans"]))
    for g in doc["spans"]:
        hit = [sp for sp in spans
               if sp["start"] < g["end"] and g["start"] < sp["end"]]
        if not hit:
            verdict = "완전미탐"
            covered = ""
        else:
            lo = max(g["start"], min(h["start"] for h in hit))
            hi = min(g["end"], max(h["end"] for h in hit))
            covered = flat(text[lo:hi])
            verdict = "완전탐지" if (lo <= g["start"] and hi >= g["end"]) else "부분노출"
        print("     [%4d,%4d) %-14s %-8s 정답=%-24r 덮인구간=%r"
              % (g["start"], g["end"], g["corp_category"], verdict,
                 flat(g["value"])[:24], covered[:24]))
    print()
    print("  [마스킹 결과]")
    print("   ", flat(masked)[:600])
    print()


def main():
    ap = argparse.ArgumentParser(prog="opf_probe.py")
    ap.add_argument("--ckpt", default=opf_local.DEFAULT_CKPT)
    ap.add_argument("--device", default=None, help="local 백엔드 device (기본 자동)")
    ap.add_argument("--text", default=None)
    ap.add_argument("--doc-ids", nargs="*", default=None)
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--bench", type=int, default=0, help="앞 N건으로 속도 실측")
    ap.add_argument("--repeat", type=int, default=0, help="동일 입력 N회 재현성 확인")
    args = ap.parse_args()

    t0 = time.time()
    tokenizer, backend, id2label = opf_local.build(
        "local", args.ckpt, args.device)
    load_s = time.time() - t0
    dev = getattr(backend, "device", "-")
    print("백엔드 %s / device %s / 모델 로드 %.1f초" % (backend.name, dev, load_s))
    print()

    if args.text:
        enc, tags, _ = opf_local.tag_tokens(args.text, tokenizer, backend, id2label)
        print("=" * 100)
        print("입력: %r" % args.text)
        print("토큰 수 %d == offsets %d  → 일치 검증 통과"
              % (len(enc.ids), len(enc.offsets)))
        print("=" * 100)
        dump_tokens(args.text, enc, tags)
        spans = opf_local.group_spans(tags, enc.offsets)
        print()
        print("  spans  :", json.dumps(spans, ensure_ascii=False))
        print("  masked :", opf_local.mask_text(args.text, spans))
        print()

    if args.doc_ids:
        docs = load_gold(args.gold)
        for i in args.doc_ids:
            if i not in docs:
                print("!! %s 없음" % i)
                continue
            show_doc(docs[i], tokenizer, backend, id2label)

    if args.repeat:
        docs = load_gold(args.gold)
        target = args.text or docs["SS-KO-00001"]["text"]
        sigs = []
        for r in range(args.repeat):
            enc, tags, logits = opf_local.tag_tokens(
                target, tokenizer, backend, id2label)
            spans = opf_local.group_spans(tags, enc.offsets)
            sigs.append(json.dumps(spans, ensure_ascii=False, sort_keys=True))
            print("  %d회차: 토큰 %d / 스팬 %d / 태그해시 %s"
                  % (r + 1, len(enc.ids), len(spans), hash(tuple(tags)) & 0xFFFFFFFF))
        print("  → %d회 전부 동일: %s" % (args.repeat, len(set(sigs)) == 1))
        print()

    if args.bench:
        docs = load_gold(args.gold)
        sel = list(docs.values())[: args.bench]
        n_tok = 0
        t0 = time.time()
        for d in sel:
            enc, tags, _ = opf_local.tag_tokens(
                d["text"], tokenizer, backend, id2label)
            n_tok += len(enc.ids)
        el = time.time() - t0
        per = el / len(sel)
        print("  실측 %d건 / %.2f초 / 토큰 %d" % (len(sel), el, n_tok))
        print("  건당 %.3f초 · 초당 %.1f건 · 초당 %.0f토큰" % (per, 1 / per, n_tok / el))
        print("  → 3,000건 추정 %.1f분 (실측 %d건 외삽)" % (per * 3000 / 60, len(sel)))
        print()


if __name__ == "__main__":
    main()
