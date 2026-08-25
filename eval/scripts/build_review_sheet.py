#!/usr/bin/env python3
"""표본검수 300건 시트를 생성한다.

개인정보보호위원회 「가명정보 처리 가이드라인」 2026.03 별권 제7장의
"통계적 표본검수 10%" 에 대응한다. 전체 3,000문서 중 300문서를 층화 표집한다.

층 정의 (정본 meta/spans 에서 직접 판정):
  L1 고유식별정보 포함 : corp_group == '고유식별정보' 스팬이 1건 이상
  L2 난이도 케이스     : difficulty_cases 에 경칭결합/한영혼용/조사결합 중 하나 이상
  L3 과탐지 유도 주입  : difficulty_cases 에 '과탐지유도' (주문번호·종목코드, 정답 스팬 없음)
  L4 금융 문맥         : meta.finance_context == True
  L5 일반              : 위 어디에도 안 드는 문서

층은 서로 겹치므로 문서마다 하나의 층으로 배타 배정한다. 배정 우선순위는
**모집단이 작은 층 우선**이며 순위는 데이터에서 계산한다(하드코딩하지 않는다).
L1 우선 배정은 쓸 수 없다 — L1 이 전체의 약 91%를 덮어 L3/L4 잔여 모집단이
30문서 미만으로 말라 최소 30건 요건을 만족할 수 없기 때문이다.

미탐/과탐 판정은 doc_level_miss.py / fp_breakdown.py 와 동일한 비대칭 규칙이다.
  미탐(재현율 축) : 정답 스팬이 어떤 예측 스팬에도 포함되지 않음 (정답 ⊆ 예측 실패)
  과탐(정밀도 축) : 예측 스팬이 어떤 정답 스팬에도 포함되지 않음 (예측 ⊆ 정답 실패)
두 축은 모집단이 다르므로 합산하지 않는다.

표준 라이브러리만 사용한다.
"""
import argparse
import csv
import json
import os
import random
import sys

TEXT_KEYS = ("text", "input_text", "input", "content", "document")
SPAN_KEYS = ("predictions", "predicted_spans", "pred_spans", "prediction",
             "spans", "entities", "pred")

STRATA = ("L1", "L2", "L3", "L4", "L5")
STRATA_DESC = {
    "L1": "고유식별정보 포함",
    "L2": "난이도 케이스(경칭결합·한영혼용·조사결합)",
    "L3": "과탐지 유도 주입",
    "L4": "금융 문맥",
    "L5": "일반",
}
DIFF_CASES = ("경칭결합", "한영혼용", "조사결합")
FP_MARKER = "과탐지유도"

CRITERIA_LINES = [
    "판정값은 정탐/미탐/과탐/판단보류 중 하나",
    "미탐이 1건이라도 있으면 그 문서는 미탐 문서로 집계",
    "고유식별정보 미탐은 사유를 반드시 기재",
    "판단보류는 2인 교차 검토 대상",
]

CELL_MAX_ITEMS = 5


def die(msg, code=2):
    sys.stderr.write("[build_review_sheet] %s\n" % msg)
    raise SystemExit(code)


def need_file(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))


# ── 입력 로딩 ────────────────────────────────────────────────
def load_gold(path):
    need_file(path, "정본 JSON")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        die("정본 JSON 파싱 실패: %s" % e)
    docs = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(docs, list) or not docs:
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % path)
    return docs


def parse_spans(raw):
    """예측 스팬을 (start, end, label) 리스트로 정규화한다."""
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
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
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
            recs.append({"text": text, "spans": parse_spans(raw)})
    if bad:
        sys.stderr.write("[경고] JSON 파싱 실패 라인 %d 건은 건너뜀\n" % bad)
    if not recs:
        die("predictions JSONL 에 레코드가 없습니다: %s" % path)
    if all(r["text"] is None for r in recs):
        die("predictions JSONL 에서 text 필드를 찾지 못했습니다. "
            "지원 키: %s" % ", ".join(TEXT_KEYS))
    return recs


# ── 층 판정 ──────────────────────────────────────────────────
def membership(doc):
    md = doc.get("meta") or {}
    dc = set(md.get("difficulty_cases") or [])
    return {
        "L1": any(s.get("corp_group") == "고유식별정보"
                  for s in doc.get("spans", [])),
        "L2": bool(dc.intersection(DIFF_CASES)),
        "L3": FP_MARKER in dc,
        "L4": bool(md.get("finance_context")),
    }


def assign_strata(docs):
    """겹치는 층을 배타 배정한다. 모집단이 작은 층부터 가져간다."""
    mem = [membership(d) for d in docs]
    raw = {k: sum(1 for m in mem if m[k]) for k in ("L1", "L2", "L3", "L4")}
    # 동률은 층 이름으로 결정적 정렬 (재현성 보장)
    order = sorted(raw, key=lambda k: (raw[k], k))
    assigned = []
    for m in mem:
        assigned.append(next((k for k in order if m[k]), "L5"))
    return assigned, raw, order


# ── 배분 계산 ────────────────────────────────────────────────
def allocate(pool_sizes, n_total, floor):
    """층별 표집 건수: 최소 floor 보장 후 잔여를 모집단 비례(최대잔여법) 배분."""
    live = [s for s in STRATA if pool_sizes[s] > 0]
    if len(live) < len(STRATA):
        die("모집단이 0인 층이 있어 최소 %d건 요건을 만족할 수 없습니다: %s"
            % (floor, ", ".join(s for s in STRATA if pool_sizes[s] == 0)))
    if floor * len(STRATA) > n_total:
        die("최소 %d건 x %d개 층 = %d 이 목표 표본 %d 건을 초과합니다."
            % (floor, len(STRATA), floor * len(STRATA), n_total))
    short = [s for s in STRATA if pool_sizes[s] < floor]
    if short:
        die("모집단이 최소 %d건에 못 미치는 층이 있습니다: %s"
            % (floor, ", ".join("%s(%d)" % (s, pool_sizes[s]) for s in short)))

    alloc = {s: floor for s in STRATA}
    rest = n_total - floor * len(STRATA)
    grand = sum(pool_sizes[s] for s in STRATA)
    quota = {s: rest * pool_sizes[s] / grand for s in STRATA}
    for s in STRATA:
        alloc[s] += int(quota[s])
    left = n_total - sum(alloc.values())
    # 최대잔여법 — 소수부 큰 순, 동률은 층 이름으로 결정적 처리
    ranked = sorted(STRATA, key=lambda s: (-(quota[s] - int(quota[s])), s))
    i = 0
    while left > 0:
        s = ranked[i % len(ranked)]
        if alloc[s] < pool_sizes[s]:
            alloc[s] += 1
            left -= 1
        i += 1
        if i > len(ranked) * n_total:
            die("잔여 배분이 수렴하지 않습니다 (모집단 부족).")
    return alloc


# ── 미탐/과탐 판정 ───────────────────────────────────────────
def find_misses(gold_spans, pred_spans):
    """미탐: 정답 ⊆ 예측 을 만족하는 예측이 하나도 없는 정답 스팬."""
    out = []
    for g in gold_spans:
        gs, ge = g["start"], g["end"]
        if not any(ps <= gs and ge <= pe for ps, pe, _ in pred_spans):
            out.append(g)
    return out


def find_fps(text, gold_spans, pred_spans):
    """과탐: 예측 ⊆ 정답 을 만족하는 정답이 하나도 없는 예측 스팬."""
    golds = [(g["start"], g["end"]) for g in gold_spans]
    out = []
    for ps, pe, lb in pred_spans:
        if not any(gs <= ps and pe <= ge for gs, ge in golds):
            out.append((lb, text[ps:pe]))
    return out


def flat(s):
    return " ".join(str(s).split())


def cell(items):
    if not items:
        return ""
    head = items[:CELL_MAX_ITEMS]
    txt = " | ".join(head)
    if len(items) > CELL_MAX_ITEMS:
        txt += " | 외 %d건" % (len(items) - CELL_MAX_ITEMS)
    return "(%d건) %s" % (len(items), txt)


# ── 메인 ─────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description="표본검수 300건 시트(층화 표집)를 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="정본 JSON 경로 (필수)")
    ap.add_argument("--predictions", default=None,
                    help="opf eval --predictions-out JSONL 경로 (선택). "
                         "생략하면 미탐/과탐 후보 칸을 비워 둔다.")
    ap.add_argument("--out", default=os.path.join(base, "results",
                                                  "review_sheet_300.csv"),
                    help="출력 CSV 경로")
    ap.add_argument("--criteria-out", default=os.path.join(base, "results",
                                                           "review_criteria.txt"),
                    help="검수 기준 TXT 경로")
    ap.add_argument("--n", type=int, default=300, help="표집 문서 수 (기본 300)")
    ap.add_argument("--min-per-stratum", type=int, default=30,
                    help="층별 최소 표집 건수 (기본 30)")
    ap.add_argument("--seed", type=int, default=20260818, help="표집 시드")
    args = ap.parse_args()

    if args.n <= 0:
        die("--n 은 1 이상이어야 합니다: %d" % args.n)
    if args.min_per_stratum < 0:
        die("--min-per-stratum 은 0 이상이어야 합니다: %d" % args.min_per_stratum)

    docs = load_gold(args.gold)
    assigned, raw, order = assign_strata(docs)

    pools = {s: [] for s in STRATA}
    for doc, s in zip(docs, assigned):
        pools[s].append(doc)
    pool_sizes = {s: len(pools[s]) for s in STRATA}

    print("")
    print("=" * 78)
    print("[0] 입력")
    print("=" * 78)
    print("정본 JSON     : %s" % args.gold)
    print("정본 문서 수  : %d" % len(docs))
    print("표집 시드     : %d" % args.seed)
    print("목표 표본 수  : %d (층별 최소 %d)" % (args.n, args.min_per_stratum))

    print("")
    print("=" * 78)
    print("[1] 층 배정 (겹침 허용 원시 모집단 -> 배타 배정 모집단)")
    print("=" * 78)
    print("배정 우선순위 : %s  (모집단 작은 층 우선, 데이터에서 산출)"
          % " > ".join(order + ["L5"]))
    hdr = "%-4s  %-34s  %-12s  %-12s" % ("층", "정의", "원시 모집단", "배타 모집단")
    print(hdr)
    print("-" * 74)
    for s in STRATA:
        print("%-4s  %-34s  %-12s  %-12s" % (
            s, STRATA_DESC[s],
            ("%d/%d" % (raw[s], len(docs))) if s in raw else "(잔차)",
            "%d/%d" % (pool_sizes[s], len(docs))))
    print("-" * 74)
    print("%-4s  %-34s  %-12s  %-12s" % (
        "합계", "", "", "%d/%d" % (sum(pool_sizes.values()), len(docs))))

    alloc = allocate(pool_sizes, args.n, args.min_per_stratum)

    # 표집 — 층 순서와 문서 순서를 고정해 시드 재현성을 보장한다
    rnd = random.Random(args.seed)
    picked = []
    for s in STRATA:
        cand = sorted(pools[s], key=lambda d: d.get("id"))
        picked.extend((d, s) for d in rnd.sample(cand, alloc[s]))
    picked.sort(key=lambda ds: ds[0].get("id"))

    print("")
    print("=" * 78)
    print("[2] 층별 실제 배분 건수")
    print("=" * 78)
    hdr = "%-4s  %-34s  %-12s  %-10s  %-8s" % (
        "층", "정의", "모집단", "배분", "최소%d" % args.min_per_stratum)
    print(hdr)
    print("-" * 78)
    got = {s: 0 for s in STRATA}
    for _, s in picked:
        got[s] += 1
    ok_all = True
    for s in STRATA:
        ok = got[s] >= args.min_per_stratum
        ok_all = ok_all and ok
        print("%-4s  %-34s  %-12s  %-10s  %-8s" % (
            s, STRATA_DESC[s], "%d/%d" % (pool_sizes[s], len(docs)),
            "%d/%d" % (got[s], args.n), "충족" if ok else "미달"))
    print("-" * 78)
    print("%-4s  %-34s  %-12s  %-10s  %-8s" % (
        "합계", "", "%d" % len(docs), "%d/%d" % (sum(got.values()), args.n),
        "전층충족" if ok_all else "미달있음"))
    if not ok_all:
        die("층별 최소 %d건 요건을 만족하지 못했습니다." % args.min_per_stratum)

    # 예측 조인
    pred_by_text = None
    if args.predictions:
        preds = load_predictions(args.predictions)
        pred_by_text = {r["text"]: r for r in preds if r["text"] is not None}
        fail = [d.get("id") for d, _ in picked
                if d.get("text") not in pred_by_text]
        print("")
        print("=" * 78)
        print("[3] 예측 조인 (text 완전 일치)")
        print("=" * 78)
        print("predictions        : %s" % args.predictions)
        print("predictions 레코드 : %d" % len(preds))
        print("표본 조인 성공     : %d/%d" % (len(picked) - len(fail), len(picked)))
        print("표본 조인 실패     : %d/%d" % (len(fail), len(picked)))
        if fail:
            print("실패 문서 id (최대 20건): %s"
                  % ", ".join(str(x) for x in fail[:20]))
            print("")
            print("*** 실패: 조인 실패 건수가 0이 아닙니다. 집계를 중단합니다. ***")
            raise SystemExit(1)
        print("조인 실패 0건 — 미탐/과탐 후보를 채웁니다.")
    else:
        print("")
        print("=" * 78)
        print("[3] 예측 조인")
        print("=" * 78)
        print("predictions 미지정 — 예측스팬수/미탐_후보/과탐_후보 3개 칸을")
        print("빈 칸으로 두고 생성합니다. 내일 opf eval 산출물이 나오면")
        print("--predictions 를 붙여 같은 시드로 재생성하면 동일 300건에 값만 채워집니다.")

    # CSV 작성
    header = ["doc_id", "층", "char_len", "정답스팬수", "예측스팬수",
              "text_요약", "미탐_후보", "과탐_후보",
              "검수자판정", "사유", "검수자"]
    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
        except OSError as e:
            die("출력 디렉터리를 만들 수 없습니다: %s (%s)" % (outdir, e))

    n_miss_docs = n_fp_docs = n_miss = n_fp = 0
    try:
        f = open(args.out, "w", encoding="utf-8-sig", newline="")
    except OSError as e:
        die("출력 CSV 를 열 수 없습니다: %s (%s)" % (args.out, e))
    with f:
        w = csv.writer(f)
        w.writerow(header)
        for doc, s in picked:
            text = doc.get("text", "")
            gold = doc.get("spans", [])
            md = doc.get("meta") or {}
            n_pred, miss_cell, fp_cell = "", "", ""
            if pred_by_text is not None:
                pspans = pred_by_text[text]["spans"]
                n_pred = len(pspans)
                misses = find_misses(gold, pspans)
                fps = find_fps(text, gold, pspans)
                miss_cell = cell(["%s=%s" % (g.get("corp_category") or "(미분류)",
                                             flat(g.get("value", "")))
                                  for g in misses])
                fp_cell = cell(["%s=%s" % (lb or "(무라벨)", flat(sur))
                                for lb, sur in fps])
                n_miss += len(misses)
                n_fp += len(fps)
                n_miss_docs += 1 if misses else 0
                n_fp_docs += 1 if fps else 0
            w.writerow([doc.get("id"), s,
                        md.get("char_len", len(text)), len(gold), n_pred,
                        flat(text[:100]), miss_cell, fp_cell, "", "", ""])

    try:
        with open(args.criteria_out, "w", encoding="utf-8") as cf:
            for line in CRITERIA_LINES:
                cf.write("- %s\n" % line)
    except OSError as e:
        die("검수 기준 파일을 쓸 수 없습니다: %s (%s)" % (args.criteria_out, e))

    print("")
    print("=" * 78)
    print("[4] 산출물")
    print("=" * 78)
    print("검수 시트   : %s (헤더 1행 + %d행, UTF-8 BOM)" % (args.out, len(picked)))
    print("검수 기준   : %s (%d행)" % (args.criteria_out, len(CRITERIA_LINES)))
    if pred_by_text is not None:
        print("미탐 후보   : %d스팬 / 미탐 문서 %d/%d" % (n_miss, n_miss_docs, len(picked)))
        print("과탐 후보   : %d스팬 / 과탐 문서 %d/%d" % (n_fp, n_fp_docs, len(picked)))
        print("주의: 미탐(재현율 축)과 과탐(정밀도 축)은 모집단이 달라 합산하지 않는다.")
    else:
        print("미탐/과탐 후보 : 미기재 (predictions 미지정)")
    print("")


if __name__ == "__main__":
    main()
