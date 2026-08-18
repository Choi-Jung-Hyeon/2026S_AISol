#!/usr/bin/env python3
"""OPF 단독 / 규칙 단독 / 합집합 3단 비교.

도입 판단에 직접 쓰이는 표다. 규칙 레이어가 OPF 미탐을 몇 건 구제하는지,
그 대가로 과탐이 얼마나 늘어나는지를 같은 축에서 본다.

재현율 매칭은 "정답 ⊆ 예측" 축을 쓴다(doc_level_miss.py 와 동일 규칙).
과탐은 "예측 ⊆ 정답" 을 만족하지 못한 예측이며 정밀도 축이다.
두 축의 TP 개수는 다르므로 섞지 않고, FN 과 FP 를 합산한 값은 만들지 않는다.

조인은 text 완전 일치로만 한다. predictions JSONL 의 example_id 는
sha256 자동 생성값이라 정본 id 와 다르기 때문이다.

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
    sys.stderr.write("[hybrid_merge] %s\n" % msg)
    raise SystemExit(2)


def parse_spans(raw):
    out = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        for key, offs in raw.items():
            label = key.split(":", 1)[0].strip() if ":" in key else key
            for off in offs or []:
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


def load_predictions(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))
    by_text = {}
    n = 0
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
            if text is None:
                continue
            raw = None
            for k in SPAN_KEYS:
                if k in r:
                    raw = r[k]
                    break
            by_text[text] = parse_spans(raw)
            n += 1
    if not n:
        die("%s 에 유효한 레코드가 없습니다: %s" % (what, path))
    return by_text


def load_gold(path):
    if not os.path.isfile(path):
        die("정본 JSON 파일을 찾을 수 없습니다: %s" % path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        die("정본 JSON 파싱 실패: %s (%s)" % (path, e))
    docs = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(docs, list):
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % path)
    return docs


def merge_spans(a, b):
    """두 예측 집합을 합친다. 겹치면 넓은 쪽을 채택한다.

    돌려주는 값: (병합 결과, 겹쳐서 흡수된 건수)
    """
    allspans = sorted(set(a) | set(b), key=lambda x: (x[0], -(x[1] - x[0])))
    out = []
    absorbed = 0
    for s, e, lb in allspans:
        placed = False
        for i, (os_, oe, olb) in enumerate(out):
            if s < oe and os_ < e:           # 겹침
                if (e - s) > (oe - os_):     # 새 쪽이 더 넓다
                    out[i] = (s, e, lb)
                absorbed += 1
                placed = True
                break
        if not placed:
            out.append((s, e, lb))
    return out, absorbed


def covered(g, preds):
    """재현율 축: 정답 스팬 ⊆ 예측 스팬"""
    return any(s <= g["start"] and g["end"] <= e for s, e, _ in preds)


def is_fp(p, golds):
    """정밀도 축: 예측 ⊆ 정답 을 만족하지 못하면 과탐"""
    s, e, _ = p
    return not any(g["start"] <= s and e <= g["end"] for g in golds)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description="OPF 단독 / 규칙 단독 / 합집합 3단 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=os.path.join(
        root, "data", "ss_pii_testset_ko_v1.json"), help="정본 JSON 경로")
    ap.add_argument("--opf", required=True,
                    help="opf eval --predictions-out JSONL 경로")
    ap.add_argument("--rule", default=os.path.join(
        root, "results", "rule_predictions.jsonl"),
        help="rule_layer.py 가 만든 예측 JSONL 경로")
    ap.add_argument("--focus", default="고유식별정보",
                    choices=["고유식별정보", "전체"],
                    help="항목 표의 범위 (기본: 고유식별정보 4종)")
    args = ap.parse_args()

    docs = load_gold(args.gold)
    opf_by_text = load_predictions(args.opf, "OPF predictions")
    rule_by_text = load_predictions(args.rule, "규칙 레이어 predictions")

    UNIQ4 = ["주민등록번호", "외국인등록번호", "여권번호", "운전면허번호"]

    join_fail_opf = join_fail_rule = 0
    pairs = []
    for doc in docs:
        t = doc.get("text")
        o = opf_by_text.get(t)
        r = rule_by_text.get(t)
        if o is None:
            join_fail_opf += 1
        if r is None:
            join_fail_rule += 1
        if o is None or r is None:
            continue
        pairs.append((doc, o, r))

    print("")
    print("=" * 78)
    print("[0] 조인 결과 (text 완전 일치)")
    print("=" * 78)
    print("정본 문서 수      : %d" % len(docs))
    print("OPF 조인 실패     : %d/%d" % (join_fail_opf, len(docs)))
    print("규칙 조인 실패    : %d/%d" % (join_fail_rule, len(docs)))
    print("3자 조인 성공     : %d/%d" % (len(pairs), len(docs)))
    if join_fail_opf or join_fail_rule:
        print("")
        print("*** 실패: 조인 실패 건수가 0이 아닙니다. 집계를 중단합니다. ***")
        raise SystemExit(1)

    cats = UNIQ4 if args.focus == "고유식별정보" else None
    gold_n, hit = {}, {"opf": {}, "rule": {}, "union": {}}
    docs_missed = {"opf": set(), "rule": set(), "union": set()}
    fp = {"opf": 0, "rule": 0, "union": 0}
    rescued = 0        # 합집합에서만 잡힌 정답 스팬
    rescued_by_cat = {}
    absorbed_total = 0
    pred_n = {"opf": 0, "rule": 0, "union": 0}

    for doc, o, r in pairs:
        u, absorbed = merge_spans(o, r)
        absorbed_total += absorbed
        pred_n["opf"] += len(o)
        pred_n["rule"] += len(r)
        pred_n["union"] += len(u)
        golds = doc.get("spans", [])
        did = doc.get("id")

        for g in golds:
            c = g.get("corp_category")
            if cats is not None and c not in cats:
                continue
            gold_n[c] = gold_n.get(c, 0) + 1
            co = covered(g, o)
            cr = covered(g, r)
            cu = covered(g, u)
            for name, ok in (("opf", co), ("rule", cr), ("union", cu)):
                if ok:
                    hit[name][c] = hit[name].get(c, 0) + 1
                else:
                    docs_missed[name].add(did)
            if cu and not co:
                rescued += 1
                rescued_by_cat[c] = rescued_by_cat.get(c, 0) + 1

        for name, preds in (("opf", o), ("rule", r), ("union", u)):
            for p in preds:
                if is_fp(p, golds):
                    fp[name] += 1

    order = cats if cats is not None else sorted(gold_n, key=lambda c: -gold_n[c])

    print("")
    print("=" * 78)
    print("[1] 항목별 재현율 3단 비교 (정답 ⊆ 예측) — 탐지/정답")
    print("=" * 78)
    hdr = "%-14s  %-14s  %-14s  %-14s" % ("항목", "OPF 단독", "규칙 단독", "합집합")
    print(hdr)
    print("-" * len(hdr))
    tg = {"opf": 0, "rule": 0, "union": 0}
    tn = 0
    for c in order:
        g = gold_n.get(c, 0)
        if not g:
            continue
        tn += g
        row = [c]
        for name in ("opf", "rule", "union"):
            h = hit[name].get(c, 0)
            tg[name] += h
            row.append("%d/%d" % (h, g))
        print("%-14s  %-14s  %-14s  %-14s" % tuple(row))
    print("-" * len(hdr))
    print("%-14s  %-14s  %-14s  %-14s" % (
        "합계", "%d/%d" % (tg["opf"], tn), "%d/%d" % (tg["rule"], tn),
        "%d/%d" % (tg["union"], tn)))

    print("")
    print("=" * 78)
    print("[2] 항목별 미탐 건수 3단 비교 — 미탐/정답")
    print("=" * 78)
    print(hdr)
    print("-" * len(hdr))
    for c in order:
        g = gold_n.get(c, 0)
        if not g:
            continue
        row = [c] + ["%d/%d" % (g - hit[name].get(c, 0), g)
                     for name in ("opf", "rule", "union")]
        print("%-14s  %-14s  %-14s  %-14s" % tuple(row))
    print("-" * len(hdr))
    print("%-14s  %-14s  %-14s  %-14s" % (
        "합계", "%d/%d" % (tn - tg["opf"], tn), "%d/%d" % (tn - tg["rule"], tn),
        "%d/%d" % (tn - tg["union"], tn)))

    print("")
    print("=" * 78)
    print("[3] 규칙 레이어가 구제한 미탐 (합집합에서만 잡힌 정답 스팬)")
    print("=" * 78)
    print("구제 건수 : %d/%d" % (rescued, tn))
    if rescued_by_cat:
        for c in order:
            if rescued_by_cat.get(c):
                print("  %-14s %d/%d" % (c, rescued_by_cat[c], gold_n.get(c, 0)))
    else:
        print("  항목별 구제 없음")

    print("")
    print("=" * 78)
    print("[4] 과탐 및 병합")
    print("=" * 78)
    print("예측 스팬 수    OPF %d / 규칙 %d / 합집합 %d" % (
        pred_n["opf"], pred_n["rule"], pred_n["union"]))
    print("과탐(FP) 건수   OPF %d / 규칙 %d / 합집합 %d" % (
        fp["opf"], fp["rule"], fp["union"]))
    delta = fp["union"] - fp["opf"]
    print("합집합 과탐 증가분 (합집합 - OPF 단독) : %d" % delta)
    print("겹쳐서 병합된 예측 건수                : %d" % absorbed_total)
    print("  (겹치는 예측은 넓은 쪽을 채택했다)")
    if delta < 0:
        print("  증가분이 음수인 이유: 서로 겹치는 OPF 과탐 여러 건이 병합되어")
        print("  한 건으로 합쳐지면 합집합의 과탐 건수가 OPF 단독보다 줄어든다.")
        print("  과탐이 실제로 사라진 것이 아니라 스팬 수 세는 단위가 바뀐 것이다.")

    print("")
    print("=" * 78)
    print("[5] 문서 단위 미탐율 3단 비교")
    print("=" * 78)
    n = len(pairs)
    hdr2 = "%-14s  %-14s  %-14s  %-14s" % ("구분", "OPF 단독", "규칙 단독", "합집합")
    print(hdr2)
    print("-" * len(hdr2))
    print("%-14s  %-14s  %-14s  %-14s" % (
        "미탐 문서",
        "%d/%d" % (len(docs_missed["opf"]), n),
        "%d/%d" % (len(docs_missed["rule"]), n),
        "%d/%d" % (len(docs_missed["union"]), n)))
    print("%-14s  %-14s  %-14s  %-14s" % (
        "전부 탐지",
        "%d/%d" % (n - len(docs_missed["opf"]), n),
        "%d/%d" % (n - len(docs_missed["rule"]), n),
        "%d/%d" % (n - len(docs_missed["union"]), n)))
    print("")
    print("범위: %s" % ("고유식별정보 4종" if cats else "전체 항목"))
    print("")


if __name__ == "__main__":
    main()
