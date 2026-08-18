#!/usr/bin/env python3
"""과탐(FP)을 3버킷으로 분해한다.

FP 의 정의는 정밀도 축이다: '예측 스팬 ⊆ 정답 스팬' 을 만족하지 못한 예측.
재현율 축 TP(정답 ⊆ 예측)와는 개수가 다르므로 두 축을 섞지 않는다.
FN 과 FP 를 합산한 값은 만들지 않는다.

버킷 판정 순서 (앞선 조건이 우선):
  B2 업무영향 과탐 : 주문번호/종목코드 정규식에 결정적으로 걸린 예측
  B3 순수 과탐     : 정답 스팬과 부분 겹침이 있으나 경계가 어긋난 예측
  B1 설계상 과탐   : 위 둘 다 아닌 잔차(정답 스팬과 무관). 날짜/나이/성별
                     패턴 사전으로 1차 하위분류만 시도한다.
표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import re
import sys

TEXT_KEYS = ("text", "input_text", "input", "content", "document")
SPAN_KEYS = ("predictions", "predicted_spans", "pred_spans", "prediction",
             "spans", "entities", "pred")

# B2: 업무영향 과탐 — 결정적 식별
B2_PATTERNS = [
    ("주문번호", re.compile(r"\d{8}-\d{6}")),
    ("종목코드", re.compile(r"(005930|000660|035420|207940|051910)")),
]

# B1 1차 하위분류용 패턴 사전 (확정 분류가 아니라 진단용 힌트)
B1_HINTS = [
    ("날짜", re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}[-./]\d{1,2}[-./]\d{1,2}"
        r"|\d{4}\s*년|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}\s*[월]\s*\d{4})")),
    ("나이", re.compile(r"(만\s*)?\d{1,3}\s*(세|살)|^\d{1,3}$")),
    ("성별", re.compile(r"(남성|여성|남자|여자|남|여)$")),
]


def die(msg):
    sys.stderr.write("[fp_breakdown] %s\n" % msg)
    raise SystemExit(2)


def need_file(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))


def parse_spans(raw):
    out = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        for key, offsets in raw.items():
            label = key.split(":", 1)[0].strip() if ":" in key else key
            for off in offsets or []:
                if isinstance(off, (list, tuple)) and len(off) >= 2:
                    out.append((int(off[0]), int(off[1]), label))
        return out
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            s = item.get("start", item.get("start_offset", item.get("begin")))
            e = item.get("end", item.get("end_offset"))
            lb = (item.get("label") or item.get("type") or item.get("entity")
                  or item.get("entity_type") or item.get("class") or "")
            if s is not None and e is not None:
                out.append((int(s), int(e), str(lb)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lb = str(item[2]) if len(item) > 2 else ""
            out.append((int(item[0]), int(item[1]), lb))
    return out


def load_predictions(path):
    need_file(path, "predictions JSONL")
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = None
            for k in TEXT_KEYS:
                if isinstance(r.get(k), str):
                    text = r[k]
                    break
            raw = None
            for k in SPAN_KEYS:
                if k in r:
                    raw = r[k]
                    break
            recs.append({"text": text, "spans": parse_spans(raw)})
    if not recs:
        die("predictions JSONL 에 레코드가 없습니다: %s" % path)
    if all(r["text"] is None for r in recs):
        die("predictions JSONL 에서 text 필드를 찾지 못했습니다. "
            "지원 키: %s" % ", ".join(TEXT_KEYS))
    return recs


def load_gold(path):
    need_file(path, "정본 JSON")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        die("정본 JSON 파싱 실패: %s" % e)
    docs = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(docs, list):
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % path)
    return docs


def classify_b1(surface):
    s = surface.strip()
    for name, pat in B1_HINTS:
        if pat.search(s):
            return name
    return "미분류"


def main():
    ap = argparse.ArgumentParser(
        description="과탐(FP)을 B1/B2/B3 3버킷으로 분해",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="정본 JSON 경로")
    ap.add_argument("--predictions", required=True,
                    help="opf eval --predictions-out JSONL 경로")
    ap.add_argument("--sample", type=int, default=10,
                    help="버킷별 샘플 출력 건수 (기본 10)")
    args = ap.parse_args()

    docs = load_gold(args.gold)
    preds = load_predictions(args.predictions)

    by_text = {r["text"]: r for r in preds if r["text"] is not None}
    join_fail = 0
    pairs = []
    for doc in docs:
        pr = by_text.get(doc.get("text"))
        if pr is None:
            join_fail += 1
            continue
        pairs.append((doc, pr["spans"]))

    print("")
    print("=" * 78)
    print("[0] 조인 결과 (text 완전 일치)")
    print("=" * 78)
    print("정본 문서 수 : %d" % len(docs))
    print("조인 성공    : %d/%d" % (len(pairs), len(docs)))
    print("조인 실패    : %d/%d" % (join_fail, len(docs)))
    if join_fail:
        print("")
        print("*** 실패: 조인 실패 건수가 0이 아닙니다. 집계를 중단합니다. ***")
        raise SystemExit(1)

    counts = {"B1": 0, "B2": 0, "B3": 0}
    doc_sets = {"B1": set(), "B2": set(), "B3": set()}
    b2_by_kind = {}
    b2_docs_by_kind = {}
    b1_by_hint = {}
    samples = {"B1": [], "B2": [], "B3": []}

    total_pred = 0
    prec_tp = 0
    for doc, pspans in pairs:
        text = doc.get("text", "")
        golds = [(g["start"], g["end"]) for g in doc.get("spans", [])]
        did = doc.get("id")
        for ps, pe, plabel in pspans:
            total_pred += 1
            # 정밀도 TP: 예측 ⊆ 정답
            if any(gs <= ps and pe <= ge for gs, ge in golds):
                prec_tp += 1
                continue
            surface = text[ps:pe]
            # B2 우선
            kind = None
            for name, pat in B2_PATTERNS:
                if pat.search(surface):
                    kind = name
                    break
            if kind:
                counts["B2"] += 1
                doc_sets["B2"].add(did)
                b2_by_kind[kind] = b2_by_kind.get(kind, 0) + 1
                b2_docs_by_kind.setdefault(kind, set()).add(did)
                if len(samples["B2"]) < args.sample:
                    samples["B2"].append((did, plabel, surface, kind))
                continue
            # B3: 정답과 부분 겹침 있으나 경계 어긋남
            if any(ps < ge and gs < pe for gs, ge in golds):
                counts["B3"] += 1
                doc_sets["B3"].add(did)
                if len(samples["B3"]) < args.sample:
                    samples["B3"].append((did, plabel, surface, "경계어긋남"))
                continue
            # B1: 잔차
            counts["B1"] += 1
            doc_sets["B1"].add(did)
            hint = classify_b1(surface)
            b1_by_hint[hint] = b1_by_hint.get(hint, 0) + 1
            if len(samples["B1"]) < args.sample:
                samples["B1"].append((did, plabel, surface, hint))

    fp_total = counts["B1"] + counts["B2"] + counts["B3"]
    print("")
    print("=" * 78)
    print("[1] 예측 모집단 (정밀도 축)")
    print("=" * 78)
    print("예측 스팬 총계          : %d" % total_pred)
    print("정밀도 TP (예측 ⊆ 정답) : %d/%d" % (prec_tp, total_pred))
    print("FP (그 외)              : %d/%d" % (fp_total, total_pred))
    print("주의: 여기의 TP 는 정밀도 축이며 재현율 축 TP(정답 ⊆ 예측)와 개수가 다르다.")

    print("")
    print("=" * 78)
    print("[2] 과탐 3버킷 분해 (건수 축 / 문서수 축)")
    print("=" * 78)
    hdr = "%-4s  %-26s  %-14s  %-10s" % (
        "버킷", "정의", "건수/FP총계", "문서수/%d" % len(docs))
    print(hdr)
    print("-" * len(hdr))
    defs = {
        "B2": "업무영향 과탐(결정적)",
        "B1": "설계상 과탐(잔차)",
        "B3": "순수 과탐(경계 어긋남)",
    }
    for b in ("B2", "B1", "B3"):
        print("%-4s  %-26s  %-14s  %-10s" % (
            b, defs[b],
            "%d/%d" % (counts[b], fp_total) if fp_total else "%d/0" % counts[b],
            "%d/%d" % (len(doc_sets[b]), len(docs))))
    print("-" * len(hdr))
    print("%-4s  %-26s  %-14s  %-10s" % (
        "합계", "", "%d/%d" % (fp_total, fp_total),
        "%d/%d" % (len(doc_sets["B1"] | doc_sets["B2"] | doc_sets["B3"]), len(docs))))

    print("")
    print("=" * 78)
    print("[3] B2 업무영향 과탐 세부 (정규식 결정적 식별)")
    print("=" * 78)
    if not b2_by_kind:
        print("B2 해당 없음")
    for kind, n in sorted(b2_by_kind.items(), key=lambda kv: -kv[1]):
        print("%-10s  건수 %d/%d   문서수 %d/%d" % (
            kind, n, fp_total if fp_total else 0,
            len(b2_docs_by_kind.get(kind, ())), len(docs)))

    print("")
    print("=" * 78)
    print("[4] B1 잔차 1차 하위분류 (날짜/나이/성별 패턴 사전, 진단용)")
    print("=" * 78)
    if not b1_by_hint:
        print("B1 해당 없음")
    for hint, n in sorted(b1_by_hint.items(), key=lambda kv: -kv[1]):
        print("%-8s  %d/%d" % (hint, n, counts["B1"]))

    print("")
    print("=" * 78)
    print("[5] 버킷별 샘플 (최대 %d건)" % args.sample)
    print("=" * 78)
    for b in ("B2", "B1", "B3"):
        print("")
        print("-- %s --" % b)
        if not samples[b]:
            print("   샘플 없음")
        for did, plabel, surface, tag in samples[b]:
            print("   [%s] %-16s %-10s %s" % (
                did, plabel or "(무라벨)", tag, surface.replace("\n", "\\n")))
    print("")


if __name__ == "__main__":
    main()
