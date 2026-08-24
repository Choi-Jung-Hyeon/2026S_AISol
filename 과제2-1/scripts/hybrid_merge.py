#!/usr/bin/env python3
"""OPF 단독 / 규칙 단독 / 합집합 비교.

도입 판단에 직접 쓰이는 표다. 규칙 레이어가 OPF 미탐을 몇 건 구제하는지,
그 대가로 과탐이 얼마나 늘어나는지를 같은 축에서 본다.

병합 규칙은 **구간 합집합 유지**다. 겹치는 두 예측은 [min(start), max(end)] 로
합치고 어느 쪽도 버리지 않는다. 이전 판은 "겹치면 넓은 쪽 채택" 이라 좁은 쪽이
덮던 문자가 통째로 사라져, 두 레이어가 어긋나게 겹칠 때 탐지 문자를 잃었다.
비교를 위해 이전 규칙(legacy)의 결과를 같은 실행에서 나란히 출력한다.

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
import unicodedata

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


def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w):
    s = str(s)
    return s + " " * max(0, w - dwidth(s))


def row5(cells, widths=(16, 22, 22, 22, 22)):
    return "  ".join(pad(c, w) for c, w in zip(cells, widths))


def pct(n, d):
    """분수와 백분율을 병기한다 — 예: 7,689 / 7,689 (100.00%)"""
    return "{:,}/{:,} ({})".format(n, d, "n/a" if not d else "%.2f%%" % (100.0 * n / d))


def merge_union(a, b):
    """구간 합집합 유지 — 겹치는 예측은 [min(start), max(end)] 로 합친다.

    어느 한쪽도 버리지 않으므로 병합 결과가 덮는 문자 집합은 두 입력의
    합집합과 정확히 같다. 라벨은 합쳐진 쪽 라벨을 순서대로 '+' 로 잇는다.
    맞닿기만 한 구간([0,5)과 [5,9))은 겹친 것이 아니므로 합치지 않는다.

    돌려주는 값: (병합 결과, 합쳐져 줄어든 스팬 수)
    """
    allspans = sorted(set(a) | set(b))
    out = []
    for s, e, lb in allspans:
        if out and s < out[-1][1]:              # 직전 구간과 겹침
            ps, pe, plb = out[-1]
            labs = [x for x in plb.split("+") if x]
            if lb and lb not in labs:
                labs.append(lb)
            out[-1] = (min(ps, s), max(pe, e), "+".join(labs))
        else:
            out.append((s, e, lb))
    return out, len(allspans) - len(out)


def merge_legacy(a, b):
    """[이전 규칙 — 비교용 보존] 겹치면 넓은 쪽을 채택하고 좁은 쪽을 버린다.

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
        description="OPF 단독 / 규칙 단독 / 합집합 비교 (병합 규칙 신규·legacy 병기)",
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
    LAYERS = ("opf", "rule", "union", "legacy")
    gold_n, hit = {}, {k: {} for k in LAYERS}
    docs_missed = {k: set() for k in LAYERS}
    fp = {k: 0 for k in LAYERS}
    pred_n = {k: 0 for k in LAYERS}
    rescued = {"union": 0, "legacy": 0}   # 합집합에서만 잡힌 정답 스팬
    rescued_by_cat = {"union": {}, "legacy": {}}
    merged_total = {"union": 0, "legacy": 0}
    # 고유식별 4종 잔여 미탐은 --focus 와 무관하게 항상 따로 센다 (B-3)
    uniq_gold = 0
    uniq_hit = {k: 0 for k in LAYERS}
    uniq_miss_by_cat = {k: {c: 0 for c in UNIQ4} for k in LAYERS}

    for doc, o, r in pairs:
        u, m_union = merge_union(o, r)
        lg, m_legacy = merge_legacy(o, r)
        merged_total["union"] += m_union
        merged_total["legacy"] += m_legacy
        layer_spans = {"opf": o, "rule": r, "union": u, "legacy": lg}
        for k in LAYERS:
            pred_n[k] += len(layer_spans[k])
        golds = doc.get("spans", [])
        did = doc.get("id")

        for g in golds:
            c = g.get("corp_category")
            cov = {k: covered(g, layer_spans[k]) for k in LAYERS}
            if c in UNIQ4:
                uniq_gold += 1
                for k in LAYERS:
                    if cov[k]:
                        uniq_hit[k] += 1
                    else:
                        uniq_miss_by_cat[k][c] += 1
            if cats is not None and c not in cats:
                continue
            gold_n[c] = gold_n.get(c, 0) + 1
            for k in LAYERS:
                if cov[k]:
                    hit[k][c] = hit[k].get(c, 0) + 1
                else:
                    docs_missed[k].add(did)
            for k in ("union", "legacy"):
                if cov[k] and not cov["opf"]:
                    rescued[k] += 1
                    rescued_by_cat[k][c] = rescued_by_cat[k].get(c, 0) + 1

        for k in LAYERS:
            for p in layer_spans[k]:
                if is_fp(p, golds):
                    fp[k] += 1

    order = cats if cats is not None else sorted(gold_n, key=lambda c: -gold_n[c])

    print("")
    print("=" * 78)
    print("[1] 항목별 재현율 4단 비교 (정답 ⊆ 예측) — 탐지/정답")
    print("=" * 78)
    print("병합 규칙: 합집합(신규) = 구간 합집합 유지 / 합집합(legacy) = 넓은 쪽 채택")
    print("")
    hdr = row5(["항목", "OPF 단독", "규칙 단독", "합집합(신규)", "합집합(legacy)"])
    print(hdr)
    print("-" * dwidth(hdr))
    tg = {k: 0 for k in LAYERS}
    tn = 0
    for c in order:
        g = gold_n.get(c, 0)
        if not g:
            continue
        tn += g
        row = [c]
        for name in LAYERS:
            h = hit[name].get(c, 0)
            tg[name] += h
            row.append(pct(h, g))
        print(row5(row))
    print("-" * dwidth(hdr))
    print(row5(["합계"] + [pct(tg[k], tn) for k in LAYERS]))

    print("")
    print("=" * 78)
    print("[2] 항목별 미탐 건수 4단 비교 — 미탐/정답")
    print("=" * 78)
    print(hdr)
    print("-" * dwidth(hdr))
    for c in order:
        g = gold_n.get(c, 0)
        if not g:
            continue
        print(row5([c] + [pct(g - hit[name].get(c, 0), g) for name in LAYERS]))
    print("-" * dwidth(hdr))
    print(row5(["합계"] + [pct(tn - tg[k], tn) for k in LAYERS]))

    print("")
    print("=" * 78)
    print("[3] 규칙 레이어가 구제한 미탐 (합집합에서만 잡힌 정답 스팬)")
    print("=" * 78)
    print("구제 건수  신규 %s / legacy %s"
          % (pct(rescued["union"], tn), pct(rescued["legacy"], tn)))
    if rescued_by_cat["union"] or rescued_by_cat["legacy"]:
        print("")
        print(row5(["항목", "신규", "legacy", "", ""]))
        for c in order:
            if rescued_by_cat["union"].get(c) or rescued_by_cat["legacy"].get(c):
                g = gold_n.get(c, 0)
                print(row5([c, pct(rescued_by_cat["union"].get(c, 0), g),
                            pct(rescued_by_cat["legacy"].get(c, 0), g), "", ""]))
    else:
        print("  항목별 구제 없음")

    print("")
    print("=" * 78)
    print("[4] 과탐 및 병합 — 규칙 변경 전/후")
    print("=" * 78)
    print(row5(["구분", "OPF 단독", "규칙 단독", "합집합(신규)", "합집합(legacy)"]))
    print("-" * dwidth(hdr))
    print(row5(["예측 스팬 수"] + ["{:,}".format(pred_n[k]) for k in LAYERS]))
    print(row5(["과탐(FP) 건수"] + ["{:,}".format(fp[k]) for k in LAYERS]))
    print("")
    print("합집합 과탐 증가분 (합집합 - OPF 단독) : 신규 %+d / legacy %+d"
          % (fp["union"] - fp["opf"], fp["legacy"] - fp["opf"]))
    print("겹쳐서 줄어든 스팬 수                  : 신규 {:,} / legacy {:,}".format(
        merged_total["union"], merged_total["legacy"]))
    print("")
    print("신규  : 겹치는 구간을 [min(start), max(end)] 로 합친다. 어느 쪽도 버리지 않으므로")
    print("        병합 결과가 덮는 문자 집합은 두 레이어 예측의 합집합과 정확히 같다.")
    print("legacy: 겹치면 넓은 쪽만 남기고 좁은 쪽을 버렸다. 두 예측이 어긋나게 겹치면")
    print("        좁은 쪽이 덮던 문자가 사라져 탐지가 후퇴할 수 있었다.")
    if fp["union"] - fp["opf"] < 0 or fp["legacy"] - fp["opf"] < 0:
        print("증가분이 음수인 이유: 서로 겹치는 OPF 과탐 여러 건이 한 건으로 합쳐지면")
        print("과탐 건수가 OPF 단독보다 줄어든다. 과탐이 사라진 것이 아니라 세는 단위가")
        print("바뀐 것이다. 문자 기준 과잉 마스킹은 standalone_metrics [3-8] 에서 본다.")

    print("")
    print("=" * 78)
    print("[5] 문서 단위 미탐율 4단 비교")
    print("=" * 78)
    n = len(pairs)
    hdr2 = row5(["구분", "OPF 단독", "규칙 단독", "합집합(신규)", "합집합(legacy)"])
    print(hdr2)
    print("-" * dwidth(hdr2))
    print(row5(["미탐 문서"] + [pct(len(docs_missed[k]), n) for k in LAYERS]))
    print(row5(["전부 탐지"] + [pct(n - len(docs_missed[k]), n) for k in LAYERS]))
    print("")
    print("범위: %s" % ("고유식별정보 4종" if cats else "전체 항목"))

    # ── [6] 고유식별 4종 잔여 미탐 — 병합 규칙 변경 전/후 ──
    print("")
    print("=" * 78)
    print("[6] 고유식별 4종 잔여 미탐 — 병합 규칙 변경 전/후")
    print("=" * 78)
    print("--focus 와 무관하게 항상 4종 전체를 센다. 재현율 축은 정답 ⊆ 예측 이다.")
    print("")
    print(row5(["항목", "OPF 단독", "규칙 단독", "합집합(신규)", "합집합(legacy)"]))
    print("-" * dwidth(hdr))
    for c in UNIQ4:
        print(row5([c] + ["{:,}".format(uniq_miss_by_cat[k][c]) for k in LAYERS]))
    print("-" * dwidth(hdr))
    print(row5(["잔여 미탐 합계"]
               + ["{:,}".format(uniq_gold - uniq_hit[k]) for k in LAYERS]))
    print(row5(["탐지"] + [pct(uniq_hit[k], uniq_gold) for k in LAYERS]))
    print("")
    before = uniq_gold - uniq_hit["legacy"]
    after = uniq_gold - uniq_hit["union"]
    print("변경 전 (넓은 쪽 채택) 잔여 미탐 : %d건" % before)
    print("변경 후 (구간 합집합)  잔여 미탐 : %d건" % after)
    print("차이                             : %+d건%s"
          % (after - before,
             "" if after != before else "  (이 입력에서는 두 규칙의 결과가 같다)"))
    print("")


if __name__ == "__main__":
    main()
