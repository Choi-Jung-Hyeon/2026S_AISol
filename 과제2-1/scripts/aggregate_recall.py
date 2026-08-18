#!/usr/bin/env python3
"""opf eval --metrics-out JSON(typed/untyped) 2개를 읽어 재현율 중심 표를 출력한다.

우선순위: Recall > F2 > F1 > Precision. Accuracy 는 주지표에서 제외한다.
수치는 백분율을 쓰지 않는다. 재현율/정밀도는 분수(n/d)로, F1/F2 는
분수로 환원되지 않으므로 0~1 비율 그대로 표기한다.
표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import sys

CORP_11 = [
    "주민등록번호", "외국인등록번호", "여권번호", "운전면허번호",
    "국문 성명", "영문 성명", "연락처", "계좌번호",
    "이메일 주소", "주소", "카드번호",
]
UNIQUE_ID_4 = ["주민등록번호", "외국인등록번호", "여권번호", "운전면허번호"]
OPF_5 = [
    "account_number", "private_address", "private_email",
    "private_person", "private_phone",
]


def die(msg):
    sys.stderr.write("[aggregate_recall] %s\n" % msg)
    raise SystemExit(2)


def load_json(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        die("%s 파일이 올바른 JSON 이 아닙니다: %s (%s)" % (what, path, e))


def dig(obj, path):
    """'a.b.c' 를 중첩 dict 또는 평탄화된 점 표기 키 양쪽에서 조회한다."""
    cur = obj
    for i, key in enumerate(path):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
            continue
        if isinstance(cur, dict):
            flat = ".".join(path[i:])
            if flat in cur:
                return cur[flat]
        return None
    return cur


def as_fraction(rate, support):
    """비율과 모집단 크기로 n/d 분수 문자열을 만든다.

    하니스가 주는 값은 반올림된 비율이므로 여기서 되살린 건수는 근사치다.
    '~' 표시는 rate*support 가 정수에서 0.05 이상 떨어져 복원이 불안정함을 뜻한다.
    정확한 미탐 건수는 doc_level_miss.py 가 예측 스팬에서 직접 센다.
    """
    if rate is None:
        return "n/a"
    if support is None:
        return "%.4f" % rate  # 모집단을 모르면 분수로 환원 불가
    exact = rate * support
    n = int(round(exact))
    mark = "~" if abs(exact - n) > 0.05 else ""
    return "%s%d/%d" % (mark, n, support)


def ratio(v):
    return "n/a" if v is None else "%.4f" % v


def gold_supports(gold_path):
    """정본 JSON 에서 항목별/라벨별 정답 스팬 수를 센다. 없으면 None."""
    if not gold_path:
        return None, None
    if not os.path.isfile(gold_path):
        die("정본 JSON 을 찾을 수 없습니다: %s" % gold_path)
    data = load_json(gold_path, "정본 JSON")
    docs = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(docs, list):
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % gold_path)
    corp, opf = {}, {}
    for doc in docs:
        for s in doc.get("spans", []):
            c = s.get("corp_category")
            o = s.get("opf_label")
            if c:
                corp[c] = corp.get(c, 0) + 1
            if o:
                opf[o] = opf.get(o, 0) + 1
    return corp, opf


def table(title, headers, rows, note=None):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join("%-" + str(w) + "s" for w in widths)
    print(fmt % tuple(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt % tuple(str(c) for c in r))
    if note:
        print("")
        print(note)


def main():
    ap = argparse.ArgumentParser(
        description="opf eval metrics JSON(typed/untyped)에서 재현율 중심 표를 출력",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--typed", required=True,
                    help="typed 모드 --metrics-out JSON 경로")
    ap.add_argument("--untyped", required=True,
                    help="untyped 모드 --metrics-out JSON 경로")
    ap.add_argument("--gold", default=None,
                    help="정본 JSON 경로(선택). 주면 비율을 분수로 환원한다")
    args = ap.parse_args()

    typed = load_json(args.typed, "typed metrics")
    untyped = load_json(args.untyped, "untyped metrics")
    corp_sup, opf_sup = gold_supports(args.gold)

    # --- 1. 사내 11항목별 재현율 (untyped ground_truth_label_recall) ---
    rows = []
    missing = []
    for label in CORP_11:
        r = dig(untyped, ["ground_truth_label_recall", "recall", label])
        if r is None:
            missing.append(label)
        sup = corp_sup.get(label) if corp_sup else None
        rows.append([label, as_fraction(r, sup), ratio(r)])
    note = "출처: untyped metrics 의 ground_truth_label_recall.recall.<label>"
    if missing:
        note += "\n키 없음(n/a): " + ", ".join(missing)
    if corp_sup is None:
        note += "\n--gold 미지정 → 분수 환원 불가, 비율만 표기"
    table("[1] 사내 11항목별 재현율 (untyped)",
          ["항목", "재현율(분수)", "재현율(비율)"], rows, note)

    # --- 2. OPF 5라벨별 span 지표 (typed by_class.*.span) ---
    rows = []
    for label in OPF_5:
        base = dig(typed, ["by_class", label, "span"]) or {}
        rec = base.get("recall")
        prec = base.get("precision")
        sup = opf_sup.get(label) if opf_sup else None
        rows.append([
            label,
            as_fraction(rec, sup),
            ratio(base.get("f2")),
            ratio(base.get("f1")),
            ratio(prec),
        ])
    table("[2] OPF 5라벨별 스팬 지표 (typed, by_class.<label>.span)",
          ["라벨", "recall(분수)", "f2", "f1", "precision"], rows,
          "우선순위 Recall > F2 > F1 > Precision.\n"
          "precision 은 예측 모집단이 metrics 에 없어 분수로 환원하지 않는다.\n"
          "f1/f2 는 조화평균이라 분수로 환원되지 않으므로 0~1 비율로 표기한다.")

    # --- 3. 고유식별정보 4종 미탐 건수 ---
    rows = []
    for label in UNIQUE_ID_4:
        r = dig(untyped, ["ground_truth_label_recall", "recall", label])
        sup = corp_sup.get(label) if corp_sup else None
        if r is None or sup is None:
            rows.append([label, "n/a", "n/a"])
            continue
        hit = int(round(r * sup))
        rows.append([label, "%d/%d" % (sup - hit, sup), "%d/%d" % (hit, sup)])
    tot_sup = sum((corp_sup or {}).get(l, 0) for l in UNIQUE_ID_4) if corp_sup else None
    if tot_sup:
        tot_miss = 0
        ok = True
        for label in UNIQUE_ID_4:
            r = dig(untyped, ["ground_truth_label_recall", "recall", label])
            if r is None:
                ok = False
                break
            tot_miss += corp_sup[label] - int(round(r * corp_sup[label]))
        if ok:
            rows.append(["합계", "%d/%d" % (tot_miss, tot_sup),
                         "%d/%d" % (tot_sup - tot_miss, tot_sup)])
    table("[3] 고유식별정보 4종 미탐 건수",
          ["항목", "미탐(건/전체)", "탐지(건/전체)"], rows,
          "미탐 1건도 치명적이므로 비율이 아닌 절대 건수로 읽는다.\n"
          "단 이 건수는 반올림된 재현율에서 되살린 근사치다('~'는 복원 불안정).\n"
          "확정 수치는 doc_level_miss.py 가 예측 스팬을 직접 세어 산출한다.")
    print("")


if __name__ == "__main__":
    main()
