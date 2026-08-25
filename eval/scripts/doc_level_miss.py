#!/usr/bin/env python3
"""문서 단위 미탐율을 계산한다.

정본 JSON + opf eval --predictions-out JSONL 을 text 완전 일치로 조인한다.
predictions 의 example_id 는 sha256 자동 생성값이라 정본 id 와 다르므로
절대 id 로 조인하지 않는다. 조인 실패 건수가 0이 아니면 실패로 보고한다.

재현율 관점의 TP 는 '정답 스팬 ⊆ 예측 스팬' 이다(비대칭 포함 관계).
정밀도 TP(예측 ⊆ 정답)와는 개수가 다르며 여기서는 재현율 축만 다룬다.
표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import sys

TEXT_KEYS = ("text", "input_text", "input", "content", "document")
SPAN_KEYS = ("predictions", "predicted_spans", "pred_spans", "prediction",
             "spans", "entities", "pred")


def die(msg):
    sys.stderr.write("[doc_level_miss] %s\n" % msg)
    raise SystemExit(2)


def need_file(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))


def parse_spans(raw):
    """예측 스팬을 (start, end, label) 리스트로 정규화한다."""
    out = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        # {"label: value": [[s, e], ...]} 매핑 형태
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
    bad = 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
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
            recs.append({"line": ln, "text": text, "spans": parse_spans(raw)})
    if bad:
        sys.stderr.write("[경고] JSON 파싱 실패 라인 %d 건은 건너뜀\n" % bad)
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


def covered(gold, preds, typed_label=None):
    """정답 스팬이 어떤 예측 스팬에 포함되면 True (정답 ⊆ 예측)."""
    gs, ge = gold["start"], gold["end"]
    for ps, pe, plabel in preds:
        if ps <= gs and ge <= pe:
            if typed_label is None or plabel == typed_label:
                return True
    return False


def ctx(text, start, end, width=20):
    left = text[max(0, start - width):start].replace("\n", "\\n")
    right = text[end:end + width].replace("\n", "\\n")
    return left, right


def main():
    ap = argparse.ArgumentParser(
        description="문서 단위 미탐율과 미탐 스팬 샘플을 출력",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="정본 JSON 경로")
    ap.add_argument("--predictions", required=True,
                    help="opf eval --predictions-out JSONL 경로")
    ap.add_argument("--typed", action="store_true",
                    help="예측 라벨이 opf_label 과 일치할 때만 탐지로 인정")
    ap.add_argument("--sample", type=int, default=50,
                    help="미탐 스팬 샘플 출력 건수 (기본 50)")
    args = ap.parse_args()

    docs = load_gold(args.gold)
    preds = load_predictions(args.predictions)

    # text 완전 일치 조인 (example_id 는 신뢰하지 않는다)
    by_text = {}
    dup_text = 0
    for r in preds:
        if r["text"] is None:
            continue
        if r["text"] in by_text:
            dup_text += 1
        by_text[r["text"]] = r

    joined, join_fail = 0, []
    per_doc = []
    for doc in docs:
        pr = by_text.get(doc.get("text"))
        if pr is None:
            join_fail.append(doc.get("id"))
            continue
        joined += 1
        per_doc.append((doc, pr["spans"]))

    total_docs = len(docs)
    print("")
    print("=" * 78)
    print("[0] 조인 결과 (text 완전 일치)")
    print("=" * 78)
    print("정본 문서 수        : %d" % total_docs)
    print("predictions 레코드  : %d" % len(preds))
    print("조인 성공           : %d/%d" % (joined, total_docs))
    print("조인 실패           : %d/%d" % (len(join_fail), total_docs))
    print("predictions 중복 text: %d" % dup_text)
    if join_fail:
        print("")
        print("실패 문서 id (최대 20건): %s" % ", ".join(
            str(x) for x in join_fail[:20]))
        print("")
        print("*** 실패: 조인 실패 건수가 0이 아닙니다. 집계를 중단합니다. ***")
        raise SystemExit(1)
    print("조인 실패 0건 — 계속 진행")

    # 미탐 판정
    missed_docs = set()
    missed_by_cat = {}
    docs_by_cat = {}
    missed_samples = []
    total_gold = 0
    total_missed = 0
    for doc, pspans in per_doc:
        for g in doc.get("spans", []):
            total_gold += 1
            tl = g.get("opf_label") if args.typed else None
            if covered(g, pspans, tl):
                continue
            total_missed += 1
            cat = g.get("corp_category") or "(미분류)"
            missed_by_cat[cat] = missed_by_cat.get(cat, 0) + 1
            docs_by_cat.setdefault(cat, set()).add(doc.get("id"))
            missed_docs.add(doc.get("id"))
            if len(missed_samples) < args.sample:
                left, right = ctx(doc.get("text", ""), g["start"], g["end"])
                missed_samples.append(
                    (doc.get("id"), cat, g.get("value", ""), left, right))

    print("")
    print("=" * 78)
    print("[1] 문서 단위 미탐율")
    print("=" * 78)
    print("모드                        : %s" % ("typed" if args.typed else "untyped"))
    print("정답 스팬을 하나라도 놓친 문서 : %d/%d" % (len(missed_docs), total_docs))
    print("전부 잡은 문서               : %d/%d" % (total_docs - len(missed_docs), total_docs))
    print("미탐 스팬 총계               : %d/%d" % (total_missed, total_gold))

    print("")
    print("=" * 78)
    print("[2] 미탐 항목별 분류 (건수 / 문서수)")
    print("=" * 78)
    gold_by_cat = {}
    for doc, _ in per_doc:
        for g in doc.get("spans", []):
            c = g.get("corp_category") or "(미분류)"
            gold_by_cat[c] = gold_by_cat.get(c, 0) + 1
    hdr = "%-14s  %-14s  %-12s" % ("항목", "미탐/정답", "미탐 문서수")
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(gold_by_cat, key=lambda c: -missed_by_cat.get(c, 0)):
        print("%-14s  %-14s  %-12d" % (
            cat,
            "%d/%d" % (missed_by_cat.get(cat, 0), gold_by_cat[cat]),
            len(docs_by_cat.get(cat, ()))))

    print("")
    print("=" * 78)
    print("[3] 미탐 스팬 샘플 상위 %d건 (문서 id / 항목 / 원문값 / 앞뒤 20자)" % args.sample)
    print("=" * 78)
    if not missed_samples:
        print("미탐 스팬 없음")
    for i, (did, cat, val, left, right) in enumerate(missed_samples, 1):
        print("%3d. [%s] %s" % (i, did, cat))
        print("     값   : %s" % val)
        print("     문맥 : ...%s <<%s>> %s..." % (left, val, right))
    print("")


if __name__ == "__main__":
    main()
