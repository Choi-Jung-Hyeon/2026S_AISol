#!/usr/bin/env python3
"""OPF 평가 하니스 없이 예측 스팬 목록만으로 전 지표를 산출한다.

사내 운영계에서 `opf eval` 을 못 쓰고 별도 추론 스크립트만 주어지면
하니스가 만드는 metrics JSON 이 없어 aggregate_recall.py 가 무력화된다.
이 스크립트는 "정본 + 예측 스팬 목록" 두 개만으로 모든 지표를 다시 만든다.

주지표는 표준 스팬 매칭이다. 정답↔예측을 겹침 문자 수 내림차순 그리디로
1:1 배정해 TP 를 단일 값으로 만들고, 남은 정답이 FN, 남은 예측이 FP 다.
Recall = TP/(TP+FN), Precision = TP/(TP+FP) 로 분자가 같으므로 조화평균인
F1·F2 가 정의상 성립한다. 3기준을 같은 표에 병기한다.
  strict  : 오프셋 완전 일치 + 라벨 일치
  partial : 1자 이상 겹침 + 라벨 일치      ← 정본
  type    : 1자 이상 겹침, 라벨 무관       ← 상한 참고
겹침 기준은 경계 과확장을 TP 로 인정하므로 정밀도 바로 아래에 과잉 마스킹
문자 수를 항상 병기한다.

이전 판의 비대칭 두 축은 삭제하지 않고 이름을 바꿔 [3-1R] 에 보존한다.
  정답 피복률 (정답 ⊆ 예측) — 구 재현율 축, doc_level_miss.py 와 같은 규칙
  예측 귀속률 (예측 ⊆ 정답) — 구 정밀도 축, fp_breakdown.py 와 같은 규칙
두 값은 분자도 모집단도 달라 조화평균이 정의되지 않는다. 이 두 값으로는
F1·F2 를 산출하지 않으며, 서로 합산하지도 않는다.

노출 3분류는 위 재현율 축과 별개의 축이다(문자 피복 기준).
  완전 피복 : 예측들의 합집합이 정답 스팬 전체를 덮음
  부분 노출 : 일부만 덮음 (0 < 덮인 문자 < 정답 길이)
  완전 미탐 : 한 글자도 안 덮음
'완전 피복'과 '재현율 TP'는 다르다. 인접한 예측 2개가 나눠 덮으면 합집합은
전부 덮지만 정답 ⊆ 예측 을 만족하는 단일 예측은 없다(분할 피복). 그 차이를
[3-7] 에서 대사(對査)해 출력한다.

지표 우선순위는 Recall > F2 > F1 > Precision 이다. Accuracy 는 배경 토큰
지배로 과대평가되므로 부록으로만 적는다.
표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import sys
import unicodedata

# ── 하니스 형식에서 본 적 있는 키들 ──────────────────────────
TEXT_KEYS = ("text", "input_text", "input", "content", "document")
HARNESS_SPAN_KEYS = ("predictions", "predicted_spans", "pred_spans",
                     "prediction", "entities", "pred")
SIMPLE_SPAN_KEYS = ("spans",)
ID_KEYS = ("id", "doc_id", "example_id")

# 민감도 순위 — 높을수록 먼저. 문서 분류([3-4])에 쓴다.
SENSITIVITY = [
    ("주민등록번호", 1), ("외국인등록번호", 1), ("여권번호", 1), ("운전면허번호", 1),
    ("카드번호", 2), ("계좌번호", 2),
    ("국문 성명", 3), ("영문 성명", 3), ("연락처", 3),
    ("이메일 주소", 3), ("주소", 3),
]
SENS_RANK = dict(SENSITIVITY)
SENS_GROUP = {1: "고유식별정보 4종", 2: "금융 식별정보", 3: "일반 개인식별정보"}

UNIQUE_ID_CATS = ("주민등록번호", "외국인등록번호", "여권번호", "운전면허번호")
UNIQUE_ID_ALLOWANCE = 4  # 허용 기준: 고유식별정보 미탐 4건

DIFF_CASES = ("경칭결합", "한영혼용", "조사결합", "과탐지유도")
NO_CASE = "해당없음"
UNATTRIB = "(미귀속)"


def die(msg, code=2):
    sys.stderr.write("[standalone_metrics] %s\n" % msg)
    raise SystemExit(code)


def need_file(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))


# ── 한글 폭 보정 표 출력 ─────────────────────────────────────
def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w, right=False):
    s = str(s)
    fill = " " * max(0, w - dwidth(s))
    return fill + s if right else s + fill


def table(headers, widths, rows, aligns=None):
    aligns = aligns or [False] * len(headers)
    line = "  ".join(pad(h, w) for h, w in zip(headers, widths))
    out = [line, "-" * dwidth(line)]
    for r in rows:
        if r is None:
            out.append("-" * dwidth(line))
            continue
        out.append("  ".join(pad(c, w, a) for c, w, a in zip(r, widths, aligns)))
    return "\n".join(out)


def frac(n, d):
    return "%d/%d" % (n, d)


def ratio(n, d):
    return 0.0 if d == 0 else n / d


def fbeta(p, r, beta):
    b2 = beta * beta
    den = b2 * p + r
    return 0.0 if den == 0 else (1 + b2) * p * r / den


def section(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ── 입력 로딩 ────────────────────────────────────────────────
def load_gold(path):
    need_file(path, "정본 JSON")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        die("정본 JSON 파싱 실패: %s" % e)
    except OSError as e:
        die("정본 JSON 을 읽을 수 없습니다: %s (%s)" % (path, e))
    docs = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(docs, list) or not docs:
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % path)
    scheme = data.get("label_scheme", {}) if isinstance(data, dict) else {}
    return docs, scheme


def read_jsonl(path):
    need_file(path, "예측 JSONL")
    recs, bad = [], 0
    try:
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
                if not isinstance(r, dict):
                    bad += 1
                    continue
                recs.append(r)
    except OSError as e:
        die("예측 JSONL 을 읽을 수 없습니다: %s (%s)" % (path, e))
    if bad:
        sys.stderr.write("[경고] JSON 파싱 실패 라인 %d 건은 건너뜀\n" % bad)
    if not recs:
        die("예측 JSONL 에 레코드가 없습니다: %s" % path)
    return recs


def observed_keys(recs, limit=200):
    keys = set()
    for r in recs[:limit]:
        keys.update(r.keys())
    return sorted(keys)


def detect_format(recs):
    """auto 판별. 애매하면 감지 키 목록과 함께 실패시킨다."""
    keys = set(observed_keys(recs))
    has_harness = bool(keys.intersection(HARNESS_SPAN_KEYS))
    has_simple = bool(keys.intersection(SIMPLE_SPAN_KEYS))
    if has_harness and not has_simple:
        return "harness"
    if has_simple and not has_harness:
        return "simple"
    if has_harness and has_simple:
        die("예측 형식을 자동 판별할 수 없습니다 — harness 키와 simple 키가 함께 있습니다.\n"
            "  감지된 키 : %s\n"
            "  harness 후보 : %s\n"
            "  simple 후보  : %s\n"
            "  --pred-format 을 harness 또는 simple 로 명시하십시오."
            % (", ".join(sorted(keys)),
               ", ".join(sorted(keys.intersection(HARNESS_SPAN_KEYS))),
               ", ".join(sorted(keys.intersection(SIMPLE_SPAN_KEYS)))))
    die("예측 형식을 자동 판별할 수 없습니다 — 스팬 키를 찾지 못했습니다.\n"
        "  감지된 키 : %s\n"
        "  harness 로 인정하는 키 : %s\n"
        "  simple 로 인정하는 키  : %s\n"
        "  --pred-format 을 harness 또는 simple 로 명시하십시오."
        % (", ".join(sorted(keys)) or "(없음)",
           ", ".join(HARNESS_SPAN_KEYS), ", ".join(SIMPLE_SPAN_KEYS)))


def norm_spans(raw):
    """예측 스팬을 (start, end, label) 로 정규화한다."""
    out = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        # {"label: value": [[s, e], ...]} 매핑 형태
        for key, offsets in raw.items():
            label = key.split(":", 1)[0].strip() if ":" in key else key
            for off in offsets or []:
                if isinstance(off, (list, tuple)) and len(off) >= 2:
                    out.append((int(off[0]), int(off[1]), str(label)))
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
            out.append((int(item[0]), int(item[1]), str(lb)))
    return out


def load_pred(path, fmt):
    recs = read_jsonl(path)
    keys = observed_keys(recs)
    if fmt == "auto":
        fmt = detect_format(recs)
    span_keys = HARNESS_SPAN_KEYS if fmt == "harness" else SIMPLE_SPAN_KEYS
    if not set(keys).intersection(span_keys):
        die("--pred-format %s 로 지정했으나 스팬 키가 없습니다.\n"
            "  감지된 키 : %s\n  기대한 키 : %s"
            % (fmt, ", ".join(keys), ", ".join(span_keys)))
    out = []
    for r in recs:
        text = None
        for k in TEXT_KEYS:
            if isinstance(r.get(k), str):
                text = r[k]
                break
        rid = None
        for k in ID_KEYS:
            if r.get(k) is not None:
                rid = str(r[k])
                break
        raw = None
        for k in span_keys:
            if k in r:
                raw = r[k]
                break
        try:
            spans = norm_spans(raw)
        except (TypeError, ValueError) as e:
            die("예측 스팬 오프셋이 정수가 아닙니다 (id=%s): %s" % (rid, e))
        out.append({"id": rid, "text": text, "spans": spans})
    return out, fmt, keys


# ── 구간 유틸 ────────────────────────────────────────────────
def covered_len(gs, ge, pspans):
    """예측 합집합이 [gs, ge) 를 덮은 문자 수."""
    segs = []
    for ps, pe, _ in pspans:
        a, b = max(ps, gs), min(pe, ge)
        if a < b:
            segs.append((a, b))
    if not segs:
        return 0, []
    segs.sort()
    merged = [list(segs[0])]
    for a, b in segs[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged), [(a, b) for a, b in merged]


# ── 표준 스팬 매칭 (TP 단일 정의) ────────────────────────────
# 기존 두 축(정답⊆예측 / 예측⊆정답)은 분자가 서로 달라 F1·F2 의 조화평균이
# 정의되지 않는다. 여기서는 정답↔예측을 1:1 로 배정해 TP 를 한 값으로 만든다.
CRITERIA = ("strict", "partial", "type")
CRIT_NAME = {
    "strict": "strict  (오프셋 완전일치+라벨)",
    "partial": "partial (겹침+라벨) ← 정본",
    "type": "type    (겹침, 라벨무관) ← 상한",
}


def label_match(lb, g):
    """라벨 일치 판정.

    EM 의 --em-require-label 이 쓰는 판정(L: lb in (cat, opf))과 같은 규칙이다.
    예측 라벨이 정답의 corp_category 또는 opf_label 과 같으면 일치로 본다.
    """
    return lb in ((g.get("corp_category") or "(미분류)"),
                  (g.get("opf_label") or "(무라벨)"))


def match_standard(pairs, criterion):
    """겹침 문자 수 내림차순 그리디 1:1 배정으로 TP 를 센다.

    후보 생성 : 정답·예측이 1자 이상 겹치고(strict 는 오프셋 완전 일치)
                라벨이 일치하면 후보 쌍. type 기준은 라벨을 보지 않는다.
    배정      : 겹침 문자 수 내림차순 그리디. 동점이면
                (doc_id, gold.start, pred.start) 오름차순으로 결정적 처리.
    결과      : TP = 배정된 쌍의 수(단일 값), FN = 미배정 정답, FP = 미배정 예측.
                Recall = TP/(TP+FN), Precision = TP/(TP+FP) 로 분자가 같다.
    """
    cands = []
    for di, (doc, pspans) in enumerate(pairs):
        did = str(doc.get("id"))
        golds = doc.get("spans", [])
        for gi, g in enumerate(golds):
            gs, ge = int(g["start"]), int(g["end"])
            for pi, (ps, pe, lb) in enumerate(pspans):
                ov = min(ge, pe) - max(gs, ps)
                if ov <= 0:                       # 1자도 안 겹치면 후보 아님
                    continue
                if criterion == "strict" and not (ps == gs and pe == ge):
                    continue
                if criterion != "type" and not label_match(lb, g):
                    continue
                # 정렬 키: 겹침 내림차순 → doc_id → gold.start → pred.start
                cands.append((-ov, did, gs, ps, di, gi, pi))
    cands.sort()
    gold_matched, pred_matched, assign = set(), set(), {}
    for nov, did, gs, ps, di, gi, pi in cands:
        if (di, gi) in gold_matched or (di, pi) in pred_matched:
            continue
        gold_matched.add((di, gi))
        pred_matched.add((di, pi))
        assign[(di, pi)] = gi
    return {"tp": len(gold_matched), "gold_matched": gold_matched,
            "pred_matched": pred_matched, "assign": assign}


def fracpct(n, d):
    """분수와 백분율을 병기한다 — 예: 24,489 / 28,420 (86.17%)"""
    pct = "n/a" if not d else "%.2f%%" % (100.0 * n / d)
    return "{:,} / {:,} ({})".format(n, d, pct)


def main():
    ap = argparse.ArgumentParser(
        description="OPF 하니스 없이 예측 스팬만으로 전 지표를 산출",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="정본 JSON 경로 (필수)")
    ap.add_argument("--pred", required=True, help="예측 JSONL 경로 (필수)")
    ap.add_argument("--pred-format", default="auto",
                    choices=("auto", "harness", "simple"),
                    help="예측 파일 형식 (기본 auto)")
    ap.add_argument("--em-require-label", action="store_true",
                    help="Exact Match 에 라벨 일치까지 요구한다")
    ap.add_argument("--out-json", default=None, help="전 수치를 JSON 으로 저장")
    ap.add_argument("--partial-sample", type=int, default=20,
                    help="부분 노출 대표 사례 출력 건수 (기본 20)")
    args = ap.parse_args()

    docs, scheme = load_gold(args.gold)
    preds, fmt, pkeys = load_pred(args.pred, args.pred_format)

    corp_cats = list(scheme.get("corp_categories") or [])
    opf_labels_declared = list(scheme.get("opf_labels") or [])
    # 정본에서 항목 -> OPF 라벨 사상을 직접 만든다(하드코딩하지 않는다)
    cat2opf = {}
    for d in docs:
        for g in d.get("spans", []):
            if g.get("corp_category") and g.get("opf_label"):
                cat2opf.setdefault(g["corp_category"], g["opf_label"])
    if not corp_cats:
        corp_cats = sorted({g.get("corp_category") for d in docs
                            for g in d.get("spans", []) if g.get("corp_category")})
    opf_labels = sorted({v for v in cat2opf.values()}) or opf_labels_declared

    section("[0] 입력")
    print("정본 JSON      : %s" % args.gold)
    print("정본 문서 수   : %d" % len(docs))
    print("예측 JSONL     : %s" % args.pred)
    print("예측 레코드    : %d" % len(preds))
    print("예측 형식      : %s%s" % (fmt, " (auto 판별)" if args.pred_format == "auto" else " (지정)"))
    print("감지된 키      : %s" % ", ".join(pkeys))
    print("EM 라벨 조건   : %s" % ("라벨 일치 요구" if args.em_require_label else "오프셋만 비교"))
    print("측정 단위      : 스팬 (문자 오프셋 [start, end))")
    print("사내 항목 수   : %d" % len(corp_cats))
    print("정본 OPF 라벨  : %d종 (%s)" % (len(opf_labels), ", ".join(opf_labels)))

    # ── [1] 조인 ─────────────────────────────────────────────
    by_text, dup_text = {}, 0
    for r in preds:
        if r["text"] is None:
            continue
        if r["text"] in by_text:
            dup_text += 1
        by_text[r["text"]] = r
    join_key = "text"
    by_id = {}
    if not by_text:
        join_key = "id"
        for r in preds:
            if r["id"] is not None:
                by_id[r["id"]] = r

    section("[1] 조인 (text 완전 일치)")
    if join_key == "id":
        print("주의: 예측 파일에 text 필드가 전혀 없어 id 조인으로 대체합니다.")
        print("      하니스 example_id 는 sha256 자동 생성값이라 정본 id 와 다르므로,")
        print("      이 경로는 사내 추론 스크립트가 정본 id 를 그대로 실은 경우에만 유효합니다.")
    pairs, join_fail = [], []
    for d in docs:
        r = by_text.get(d.get("text")) if join_key == "text" else by_id.get(str(d.get("id")))
        if r is None:
            join_fail.append(d.get("id"))
            continue
        pairs.append((d, r["spans"]))
    print("조인 키           : %s" % join_key)
    print("정본 문서 수      : %d" % len(docs))
    print("예측 레코드 수    : %d" % len(preds))
    print("조인 성공         : %s" % frac(len(pairs), len(docs)))
    print("조인 실패         : %s" % frac(len(join_fail), len(docs)))
    print("예측 중복 text    : %d" % dup_text)
    if join_fail:
        print("실패 문서 id (최대 20건): %s"
              % ", ".join(str(x) for x in join_fail[:20]))
        print("")
        print("*** 실패: 조인 실패 건수가 0이 아닙니다. 집계를 중단합니다. ***")
        raise SystemExit(1)
    print("조인 실패 0건 — 계속 진행")

    # ── 표준 매칭 (3기준) ────────────────────────────────────
    # 문서 루프보다 먼저 배정을 끝낸다. 루프 안에서는 배정 결과만 참조한다.
    std = {c: match_standard(pairs, c) for c in CRITERIA}
    std_gold_matched = std["partial"]["gold_matched"]
    std_pred_matched = std["partial"]["pred_matched"]

    # ── 핵심 계산 ────────────────────────────────────────────
    def zero():
        # tp/fp : 표준 매칭(partial 기준) 단일 TP 와 미배정 예측
        # rtp/ptp : 보존용 비대칭 두 축 (정답 피복률 / 예측 귀속률)
        return {"gold": 0, "tp": 0, "fp": 0,
                "rtp": 0, "em": 0, "ptp": 0, "pred": 0,
                "full": 0, "partial": 0, "none": 0, "split": 0,
                "exposed_chars": 0, "over_chars": 0}

    by_cat = {c: zero() for c in corp_cats}
    by_opf = {}
    by_case = {c: zero() for c in list(DIFF_CASES) + [NO_CASE]}
    by_tier = {}
    tot = zero()
    unattrib_pred = 0
    doc_miss_rank = {}          # doc id -> 최고민감 미탐 항목
    docs_clean, docs_missed = 0, 0
    partial_samples = []
    over_by_cat = {}

    # 길이 3분위 경계 (문서 char_len 기준)
    lens = sorted((d.get("meta") or {}).get("char_len", len(d.get("text", "")))
                  for d, _ in pairs)
    q1 = lens[len(lens) // 3]
    q2 = lens[2 * len(lens) // 3]
    tier_of = lambda L: "T1(짧음)" if L < q1 else ("T2(중간)" if L < q2 else "T3(긺)")
    for t in ("T1(짧음)", "T2(중간)", "T3(긺)"):
        by_tier[t] = zero()

    for di, (doc, pspans) in enumerate(pairs):
        text = doc.get("text", "")
        golds = doc.get("spans", [])
        md = doc.get("meta") or {}
        cases = [c for c in DIFF_CASES if c in (md.get("difficulty_cases") or [])]
        if not cases:
            cases = [NO_CASE]
        tier = tier_of(md.get("char_len", len(text)))
        buckets_doc = [by_case[c] for c in cases] + [by_tier[tier]]

        gold_iv = [(g["start"], g["end"]) for g in golds]
        gold_chars = set()
        for a, b in gold_iv:
            gold_chars.update(range(a, b))

        # ---- 재현율 축 + 노출 3분류 + EM (정답 스팬 기준) ----
        doc_missed_cats = []
        for gi, g in enumerate(golds):
            gs, ge, cat = g["start"], g["end"], g.get("corp_category") or "(미분류)"
            opf = g.get("opf_label") or "(무라벨)"
            cb = by_cat.setdefault(cat, zero())
            ob = by_opf.setdefault(opf, zero())
            targets = [tot, cb, ob] + buckets_doc
            for t in targets:
                t["gold"] += 1

            # 재현율 TP: 정답 ⊆ 예측 (단일 예측 포함)
            rtp = any(ps <= gs and ge <= pe for ps, pe, _ in pspans)
            if rtp:
                for t in targets:
                    t["rtp"] += 1
            else:
                doc_missed_cats.append(cat)

            # 표준 TP (partial 배정 결과) — 재현율·정밀도가 공유하는 단일 TP
            if (di, gi) in std_gold_matched:
                for t in targets:
                    t["tp"] += 1

            # Exact Match: start·end 완전 일치 (옵션에 따라 라벨까지)
            em = False
            for ps, pe, lb in pspans:
                if ps == gs and pe == ge:
                    if not args.em_require_label or lb in (cat, opf):
                        em = True
                        break
            if em:
                for t in targets:
                    t["em"] += 1

            # 노출 3분류: 예측 합집합의 문자 피복 (재현율 축과 별개)
            cov, segs = covered_len(gs, ge, pspans)
            glen = ge - gs
            if cov == 0:
                key = "none"
            elif cov >= glen:
                key = "full"
            else:
                key = "partial"
            for t in targets:
                t[key] += 1
            if key == "full" and not rtp:
                for t in targets:
                    t["split"] += 1
            if key == "partial":
                for t in targets:
                    t["exposed_chars"] += glen - cov
                partial_samples.append({
                    "doc_id": doc.get("id"), "corp_category": cat,
                    "value": g.get("value", ""),
                    "covered_text": " + ".join(text[a:b] for a, b in segs),
                    "remaining_text": "".join(
                        text[i] for i in range(gs, ge)
                        if not any(a <= i < b for a, b in segs)),
                    "remaining_chars": glen - cov,
                })

        if doc_missed_cats:
            docs_missed += 1
            top = min(doc_missed_cats, key=lambda c: (SENS_RANK.get(c, 9), c))
            doc_miss_rank[doc.get("id")] = top
        else:
            docs_clean += 1

        # ---- 정밀도 축 (예측 스팬 기준) ----
        for pi, (ps, pe, lb) in enumerate(pspans):
            tot["pred"] += 1
            for t in buckets_doc:
                t["pred"] += 1
            ptp = any(gs <= ps and pe <= ge for gs, ge in gold_iv)
            # 항목 귀속: 포함하는 정답 > 최대 겹침 정답 > 라벨 사상 > 미귀속
            hit = None
            for g in golds:
                if g["start"] <= ps and pe <= g["end"]:
                    hit = g
                    break
            if hit is None:
                best_ov = 0
                for g in golds:
                    ov = min(pe, g["end"]) - max(ps, g["start"])
                    if ov > best_ov:
                        best_ov, hit = ov, g
            if hit is not None:
                cat, opf = hit.get("corp_category") or "(미분류)", hit.get("opf_label") or "(무라벨)"
            elif lb in corp_cats:
                cat, opf = lb, cat2opf.get(lb, "(무라벨)")
            elif lb in opf_labels:
                cat, opf = UNATTRIB, lb
            else:
                cat, opf = UNATTRIB, UNATTRIB
            if cat == UNATTRIB:
                unattrib_pred += 1
            by_cat.setdefault(cat, zero())["pred"] += 1
            by_opf.setdefault(opf, zero())["pred"] += 1
            if ptp:
                tot["ptp"] += 1
                by_cat[cat]["ptp"] += 1
                by_opf[opf]["ptp"] += 1
                for t in buckets_doc:
                    t["ptp"] += 1

            # 표준 FP: partial 배정에서 짝을 못 찾은 예측
            if (di, pi) not in std_pred_matched:
                tot["fp"] += 1
                by_cat[cat]["fp"] += 1
                by_opf[opf]["fp"] += 1
                for t in buckets_doc:
                    t["fp"] += 1

        # ---- 과잉 마스킹: 예측 합집합 - 정답 합집합 ----
        claimed = set()
        for ps, pe, lb in sorted(pspans):
            outside = set(range(ps, pe)) - gold_chars - claimed
            claimed.update(range(ps, pe))
            if not outside:
                continue
            hit = None
            best_ov = 0
            for g in golds:
                ov = min(pe, g["end"]) - max(ps, g["start"])
                if ov > best_ov:
                    best_ov, hit = ov, g
            cat = (hit.get("corp_category") if hit is not None
                   else (lb if lb in corp_cats else UNATTRIB))
            over_by_cat[cat] = over_by_cat.get(cat, 0) + len(outside)
            tot["over_chars"] += len(outside)
            by_cat.setdefault(cat, zero())["over_chars"] += len(outside)
            for t in buckets_doc:
                t["over_chars"] += len(outside)

    G, P = tot["gold"], tot["pred"]

    # ── [3-1] 표준 지표 ─────────────────────────────────────
    section("[3-1] 표준 지표 — strict / partial / type 3기준 병기")
    print("측정 단위: 스팬 (문자 오프셋 [start, end))")
    print("")
    print("TP 는 기준마다 단일 값이다. 정답↔예측을 겹침 문자 수 내림차순 그리디로")
    print("1:1 배정하고, 배정된 쌍이 TP, 남은 정답이 FN, 남은 예측이 FP 다.")
    print("따라서 Recall = TP/(TP+FN) 과 Precision = TP/(TP+FP) 의 분자가 같은 값이며,")
    print("조화평균인 F1·F2 가 정의상 성립한다.")
    print("동점 배정 순서 : (doc_id, gold.start, pred.start) 오름차순 — 재실행 시 동일")
    print("라벨 일치 규칙 : 예측 라벨이 정답의 corp_category 또는 opf_label 과 같음")
    print("                 (--em-require-label 이 쓰는 판정과 같은 규칙)")
    print("")
    srows = []
    for c in CRITERIA:
        tp = std[c]["tp"]
        fn, fp = G - tp, P - tp
        r, pr = ratio(tp, tp + fn), ratio(tp, tp + fp)
        srows.append([CRIT_NAME[c], "{:,}".format(tp), "{:,}".format(fn),
                      "{:,}".format(fp), "%.4f" % r, "%.4f" % pr,
                      "%.4f" % fbeta(pr, r, 1), "%.4f" % fbeta(pr, r, 2)])
    print(table(["기준", "TP", "FN", "FP", "R", "P", "F1", "F2"],
                [30, 9, 9, 9, 7, 7, 7, 7], srows,
                [False, True, True, True, True, True, True, True]))

    TP = std["partial"]["tp"]
    FN, FP = G - TP, P - TP
    R, PR = ratio(TP, TP + FN), ratio(TP, TP + FP)
    print("")
    print("정본 기준 = partial (1자 이상 겹침 + 라벨 일치)")
    print("  TP        : {:,} (Recall·Precision 이 공유하는 단일 분자)".format(TP))
    print("  재현율 R  : TP/(TP+FN) = %s = %.4f" % (fracpct(TP, TP + FN), R))
    print("  정밀도 P  : TP/(TP+FP) = %s = %.4f" % (fracpct(TP, TP + FP), PR))
    print("  └ 과잉 마스킹 : {:,}자 — 예측이 정답 밖까지 덮은 문자 수. 겹침 기준은".format(
        tot["over_chars"]))
    print("                  경계 과확장을 TP 로 인정하므로, 이 보정 없이 정밀도만 보면")
    print("                  과대 해석된다. 항목별 분포는 [3-8] 에 있다.")
    print("  F1        : %.4f" % fbeta(PR, R, 1))
    print("  F2        : %.4f  (미탐이 치명적이므로 F2 를 우선한다)" % fbeta(PR, R, 2))
    print("")
    print("불변식 검증")
    print("  TP + FN = {:,} + {:,} = {:,} / 정답 스팬 {:,} → {}".format(
        TP, FN, TP + FN, G, "일치" if TP + FN == G else "*** 불일치 ***"))
    print("  TP + FP = {:,} + {:,} = {:,} / 예측 스팬 {:,} → {}".format(
        TP, FP, TP + FP, P, "일치" if TP + FP == P else "*** 불일치 ***"))
    print("  TP(strict) {:,} ≤ TP(partial) {:,} ≤ TP(type) {:,} → {}".format(
        std["strict"]["tp"], std["partial"]["tp"], std["type"]["tp"],
        "성립" if std["strict"]["tp"] <= std["partial"]["tp"] <= std["type"]["tp"]
        else "*** 위배 ***"))

    # ── [3-1R] 보존한 비대칭 두 축 ──────────────────────────
    section("[3-1R] 참고 — 비대칭 두 축 (F1·F2 를 만들지 않는다)")
    print("이전 판의 두 축을 이름만 바꿔 보존한다. 분자가 서로 달라 조화평균이")
    print("정의되지 않으므로 이 두 값으로는 F1·F2 를 산출하지 않는다.")
    print("")
    print("정답 피복률 (정답이 예측에 포함) : %s" % fracpct(tot["rtp"], G))
    print("  └ 미피복 정답                  : %s" % fracpct(G - tot["rtp"], G))
    print("예측 귀속률 (예측이 정답에 포함) : %s" % fracpct(tot["ptp"], P))
    print("  └ 미귀속 예측                  : %s" % fracpct(P - tot["ptp"], P))
    print("")
    print("두 축은 모집단이 달라 합산하지 않는다.")
    print("표준 TP(partial) {:,} 와 피복 {:,} 이 다를 수 있는 이유:".format(
        TP, tot["rtp"]))
    print("  - 피복은 1:1 배정을 하지 않아 예측 1건이 정답 여러 건을 동시에 인정받는다")
    print("  - 피복은 라벨을 보지 않는다")

    # ── [3-2] 사내 11항목별 ─────────────────────────────────
    HDR_STD = ["항목", "정답스팬", "TP", "FN", "FP", "R", "P", "F1", "F2"]
    WID_STD = [16, 9, 8, 8, 8, 7, 7, 7, 7]
    ALN_STD = [False, True, True, True, True, True, True, True, True]

    def metric_rows(keys, store):
        """partial 기준 단일 TP 로 항목별 지표를 만든다."""
        rows = []
        for k in keys:
            s = store.get(k)
            if not s or (s["gold"] == 0 and s["pred"] == 0):
                continue
            tp, fn, fp = s["tp"], s["gold"] - s["tp"], s["fp"]
            r = ratio(tp, tp + fn)
            p = ratio(tp, tp + fp)
            rows.append([k, "{:,}".format(s["gold"]), "{:,}".format(tp),
                         "{:,}".format(fn), "{:,}".format(fp),
                         "%.4f" % r, "%.4f" % p,
                         "%.4f" % fbeta(p, r, 1), "%.4f" % fbeta(p, r, 2)])
        return rows

    def legacy_rows(keys, store):
        """보존한 두 축(피복률/귀속률). F1·F2 는 만들지 않는다."""
        rows = []
        for k in keys:
            s = store.get(k)
            if not s or (s["gold"] == 0 and s["pred"] == 0):
                continue
            rows.append([k, fracpct(s["rtp"], s["gold"]),
                         fracpct(s["ptp"], s["pred"])])
        return rows

    section("[3-2] 사내 11항목별 (partial 기준 단일 TP)")
    print("측정 단위: 스팬 (문자 오프셋 [start, end))")
    print("TP 는 정답의 항목으로, FP 는 귀속 규칙으로 집계한다 — 항목별 P 의 분모는 TP+FP 다.")
    print("")
    order = sorted([c for c in by_cat if c != UNATTRIB],
                   key=lambda c: (SENS_RANK.get(c, 9), c))
    rows = metric_rows(order, by_cat)
    rows.append(None)
    rows.append(["합계", "{:,}".format(G), "{:,}".format(TP), "{:,}".format(FN),
                 "{:,}".format(FP), "%.4f" % R, "%.4f" % PR,
                 "%.4f" % fbeta(PR, R, 1), "%.4f" % fbeta(PR, R, 2)])
    print(table(HDR_STD, WID_STD, rows, ALN_STD))
    print("")
    print("FP 귀속 규칙은 포함하는 정답 > 최대 겹침 정답 > 라벨 사상 순이며,")
    print("어디에도 못 붙인 예측 %s 는 %s 로 뺐다." % (fracpct(unattrib_pred, P), UNATTRIB))
    print("")
    print("참고 — 보존한 비대칭 두 축(항목별). 조화평균을 만들지 않는다.")
    lrows = legacy_rows(order, by_cat)
    lrows += [None, ["합계", fracpct(tot["rtp"], G), fracpct(tot["ptp"], P)]]
    print(table(["항목", "정답 피복률(정답⊆예측)", "예측 귀속률(예측⊆정답)"],
                [16, 26, 26], lrows))

    # ── [3-3] OPF 라벨별 ────────────────────────────────────
    section("[3-3] OPF 라벨별")
    pred_labels = sorted({lb for _, sp in pairs for _, _, lb in sp if lb})
    is_opf = bool(pred_labels) and set(pred_labels).issubset(set(opf_labels))
    print("정본 OPF 라벨      : %s" % ", ".join(opf_labels))
    print("예측 라벨 관측값   : %s" % (", ".join(pred_labels) if pred_labels else "(라벨 없음)"))
    print("예측이 OPF 체계인가: %s" % ("예" if is_opf else "아니오"))
    if not is_opf:
        print("→ 예측 라벨이 OPF 8라벨 체계가 아니므로 정밀도 분모는 정답 귀속으로만 잡힌다.")
        print("  재현율·EM·미탐은 정답의 opf_label 기준이라 그대로 유효하다.")
    okeys = sorted([k for k in by_opf if k != UNATTRIB])
    print("")
    print("partial 기준 단일 TP. 측정 단위: 스팬 (문자 오프셋 [start, end))")
    print("")
    print(table(["OPF 라벨"] + HDR_STD[1:], [18] + WID_STD[1:],
                metric_rows(okeys, by_opf), ALN_STD))
    print("")
    print("참고 — 보존한 비대칭 두 축(OPF 라벨별). 조화평균을 만들지 않는다.")
    print(table(["OPF 라벨", "정답 피복률(정답⊆예측)", "예측 귀속률(예측⊆정답)"],
                [18, 26, 26], legacy_rows(okeys, by_opf)))

    # ── [3-4] 문서 단위 ─────────────────────────────────────
    section("[3-4] 문서 단위")
    nd = len(pairs)
    print("완전 마스킹 문서 (미탐 0건) : %s" % frac(docs_clean, nd))
    print("미탐 포함 문서              : %s" % frac(docs_missed, nd))
    print("")
    print("미탐 문서를 그 문서에서 놓친 항목 중 최고민감 항목으로 분류한다.")
    print("민감도 순위 1 고유식별정보 4종 > 2 금융 식별정보 > 3 일반 개인식별정보")
    print("")
    cnt = {}
    for c in doc_miss_rank.values():
        cnt[c] = cnt.get(c, 0) + 1
    rows = [[SENS_GROUP.get(SENS_RANK.get(c, 9), "기타"), c, frac(n, nd)]
            for c, n in sorted(cnt.items(), key=lambda kv: (SENS_RANK.get(kv[0], 9), -kv[1], kv[0]))]
    if rows:
        rows.append(None)
        rows.append(["", "합계", frac(sum(cnt.values()), nd)])
        print(table(["민감도 그룹", "최고민감 미탐 항목", "문서수"], [20, 20, 12], rows))
    else:
        print("미탐 문서 없음 — 분류표 생략")

    # ── [3-5] 고유식별정보 4종 ──────────────────────────────
    section("[3-5] 고유식별정보 4종 합산")
    ug = sum(by_cat.get(c, zero())["gold"] for c in UNIQUE_ID_CATS)
    uh = sum(by_cat.get(c, zero())["rtp"] for c in UNIQUE_ID_CATS)
    ut = sum(by_cat.get(c, zero())["tp"] for c in UNIQUE_ID_CATS)
    rows = []
    for c in UNIQUE_ID_CATS:
        b = by_cat.get(c, zero())
        rows.append([c, fracpct(b["tp"], b["gold"]), str(b["gold"] - b["tp"]),
                     fracpct(b["rtp"], b["gold"]), str(b["gold"] - b["rtp"])])
    rows.append(None)
    rows.append(["합계", fracpct(ut, ug), str(ug - ut),
                 fracpct(uh, ug), str(ug - uh)])
    print(table(["항목", "재현율(표준 TP/partial)", "FN",
                 "정답 피복률(보존)", "미피복"], [16, 26, 7, 26, 8], rows))
    print("")
    umiss = ug - ut
    umiss_legacy = ug - uh
    print("허용 기준          : 미탐 %d건 이하" % UNIQUE_ID_ALLOWANCE)
    print("실측 FN (표준)     : %d건 → %s"
          % (umiss, "충족" if umiss <= UNIQUE_ID_ALLOWANCE else "미충족"))
    print("실측 미피복 (보존) : %d건 → %s"
          % (umiss_legacy,
             "충족" if umiss_legacy <= UNIQUE_ID_ALLOWANCE else "미충족"))
    print("두 값은 정의가 다르다 — 표준 FN 은 1:1 배정+라벨 일치, 피복은 정답⊆예측 이다.")

    # ── [3-6] Exact Match ───────────────────────────────────
    section("[3-6] Exact Match (start·end 완전 일치)")
    print("라벨 일치 조건 : %s (--em-require-label)"
          % ("켬 — 예측 라벨이 항목명 또는 opf_label 과 같아야 인정"
             if args.em_require_label else "끔 — 오프셋만 비교"))
    print("전체 EM          : %s" % fracpct(tot["em"], G))
    print("정답 피복률(보존): %s" % fracpct(tot["rtp"], G))
    print("EM ≤ 정답 피복률 : %s (완전 일치는 포함의 특수 경우이므로 성립해야 한다)"
          % ("성립" if tot["em"] <= tot["rtp"] else "*** 위배 ***"))
    print("표준 TP(partial) : %s — EM 은 그 부분집합이다: %s"
          % (fracpct(std["partial"]["tp"], G),
             "성립" if tot["em"] <= std["partial"]["tp"] else "*** 위배 ***"))
    print("")
    print(table(["항목", "정답스팬", "EM"], [16, 10, 14],
                [[c, str(by_cat[c]["gold"]), frac(by_cat[c]["em"], by_cat[c]["gold"])]
                 for c in order if by_cat.get(c, zero())["gold"]]
                + [None, ["합계", str(G), frac(tot["em"], G)]]))

    # ── [3-7] 부분 노출 ─────────────────────────────────────
    section("[3-7] 부분 노출 (발표 핵심 보조지표)")
    print("정답 스팬을 예측 합집합의 문자 피복으로 3분류한다.")
    print("이 축은 정답 피복률 축(단일 예측 포함)과도, 표준 TP 축과도 별개다.")
    print("")
    print("완전 피복 (전부 덮임)      : %s" % frac(tot["full"], G))
    print("부분 노출 (일부만 덮임)    : %s" % frac(tot["partial"], G))
    print("완전 미탐 (한 글자도 안 덮임): %s" % frac(tot["none"], G))
    s3 = tot["full"] + tot["partial"] + tot["none"]
    print("합계 검증                  : %d + %d + %d = %d / 정답 %d → %s"
          % (tot["full"], tot["partial"], tot["none"], s3, G,
             "일치" if s3 == G else "*** 불일치 ***"))
    print("노출된 문자 수 합계        : %d자 (부분 노출 스팬에서 안 덮인 문자)"
          % tot["exposed_chars"])
    print("")
    print("정답 피복률 축과의 대사:")
    print("  미피복 정답          : %s" % frac(G - tot["rtp"], G))
    print("  = 완전 미탐 %d + 부분 노출 %d + 분할 피복 %d = %d"
          % (tot["none"], tot["partial"], tot["split"],
             tot["none"] + tot["partial"] + tot["split"]))
    print("  분할 피복 = 예측 2개 이상이 나눠 덮어 합집합은 완전하나")
    print("              정답 ⊆ 예측 을 만족하는 단일 예측이 없는 경우")
    print("")
    prows = [[c, str(by_cat[c]["gold"]), frac(by_cat[c]["partial"], by_cat[c]["gold"]),
              str(by_cat[c]["exposed_chars"]), frac(by_cat[c]["none"], by_cat[c]["gold"])]
             for c in order if by_cat.get(c, zero())["gold"]]
    prows += [None, ["합계", str(G), frac(tot["partial"], G),
                     str(tot["exposed_chars"]), frac(tot["none"], G)]]
    print(table(["항목", "정답스팬", "부분 노출", "노출 문자수", "완전 미탐"],
                [16, 10, 13, 12, 13], prows))
    print("")
    print("대표 사례 %d건 (남은 문자 많은 순):" % args.partial_sample)
    partial_samples.sort(key=lambda x: (-x["remaining_chars"], str(x["doc_id"])))
    if not partial_samples:
        print("  부분 노출 없음")
    for i, s in enumerate(partial_samples[:args.partial_sample], 1):
        print("%3d. [%s] %s" % (i, s["doc_id"], s["corp_category"]))
        print("     정답값   : %s" % s["value"])
        print("     덮인구간 : %s" % (s["covered_text"] or "(없음)"))
        print("     남은문자 : %s (%d자)" % (s["remaining_text"], s["remaining_chars"]))

    # ── [3-8] 과잉 마스킹 ───────────────────────────────────
    section("[3-8] 과잉 마스킹 (예측이 정답 밖까지 덮은 문자)")
    print("문서별 예측 합집합에서 정답 합집합을 뺀 문자 수다.")
    print("총 과잉 마스킹 문자 수 : %d자" % tot["over_chars"])
    orows = [[c, str(n)] for c, n in
             sorted(over_by_cat.items(), key=lambda kv: (-kv[1], kv[0])) if n]
    if orows:
        orows += [None, ["합계", str(sum(over_by_cat.values()))]]
        print("")
        print(table(["항목(최대 겹침 귀속)", "과잉 문자수"], [24, 12], orows))
    else:
        print("과잉 마스킹 없음 — 분포표 생략")

    # ── [3-9] 난이도 케이스별 ───────────────────────────────
    section("[3-9] 난이도 케이스별")
    print("문서는 케이스를 여러 개 가질 수 있어 행이 서로 배타적이지 않다(합계 생략).")
    print("")
    crows = []
    for c in list(DIFF_CASES) + [NO_CASE]:
        s = by_case[c]
        if s["gold"] == 0 and s["pred"] == 0:
            continue
        tp, fn, fp = s["tp"], s["gold"] - s["tp"], s["fp"]
        r = ratio(tp, tp + fn)
        p = ratio(tp, tp + fp)
        crows.append([c, "{:,}".format(s["gold"]), "{:,}".format(tp),
                      "{:,}".format(fn), "{:,}".format(fp),
                      "%.4f" % r, "%.4f" % p, "%.4f" % fbeta(p, r, 2),
                      frac(s["em"], s["gold"])])
    print("partial 기준 단일 TP. 측정 단위: 스팬 (문자 오프셋 [start, end))")
    print("")
    print(table(["난이도 케이스", "정답스팬", "TP", "FN", "FP", "R", "P", "F2", "EM"],
                [14, 9, 8, 8, 8, 7, 7, 7, 13], crows,
                [False, True, True, True, True, True, True, True, False]))

    # ── [3-10] 문서 길이 계층별 ─────────────────────────────
    section("[3-10] 문서 길이 계층별 (char_len 3분위)")
    print("경계 : T1 < %d ≤ T2 < %d ≤ T3" % (q1, q2))
    print("")
    trows = []
    for t in ("T1(짧음)", "T2(중간)", "T3(긺)"):
        s = by_tier[t]
        trows.append([t, str(s["gold"]), frac(s["rtp"], s["gold"]),
                      frac(s["em"], s["gold"])])
    trows += [None, ["합계", str(G), frac(tot["rtp"], G), frac(tot["em"], G)]]
    print(table(["계층", "정답스팬", "정답 피복률(보존)", "EM"],
                [12, 10, 20, 14], trows))

    # ── 부록 ────────────────────────────────────────────────
    section("[부록] Accuracy (배경 토큰 지배로 과대평가 — 주지표 아님)")
    tot_chars = sum(len(d.get("text", "")) for d, _ in pairs)
    gold_chars_all = pred_chars_all = inter = 0
    for doc, pspans in pairs:
        gc, pc = set(), set()
        for g in doc.get("spans", []):
            gc.update(range(g["start"], g["end"]))
        for ps, pe, _ in pspans:
            pc.update(range(ps, pe))
        gold_chars_all += len(gc)
        pred_chars_all += len(pc)
        inter += len(gc & pc)
    correct = tot_chars - (gold_chars_all + pred_chars_all - 2 * inter)
    print("문자 단위 정확도 : %s" % frac(correct, tot_chars))
    print("PII 문자 비중    : %s — 배경 문자가 지배해 이 수치는 참고용이다."
          % frac(gold_chars_all, tot_chars))

    # ── JSON 저장 ───────────────────────────────────────────
    if args.out_json:
        def blk(s):
            # 표준 매칭(partial) — Recall·Precision 이 같은 TP 를 분자로 쓴다
            tp, fn, fp = s["tp"], s["gold"] - s["tp"], s["fp"]
            r, p = ratio(tp, tp + fn), ratio(tp, tp + fp)
            return {
                "gold_spans": s["gold"], "pred_spans": s["pred"],
                "tp": tp, "fn": fn, "fp": fp,
                "recall": {"num": tp, "den": tp + fn},
                "precision": {"num": tp, "den": tp + fp},
                "f1": round(fbeta(p, r, 1), 6), "f2": round(fbeta(p, r, 2), 6),
                # 보존한 비대칭 두 축 — 조화평균을 만들지 않는다
                "gold_coverage": {"num": s["rtp"], "den": s["gold"]},
                "pred_attribution": {"num": s["ptp"], "den": s["pred"]},
                "em": {"num": s["em"], "den": s["gold"]},
                "full_cover": s["full"], "partial_expose": s["partial"],
                "no_cover": s["none"], "split_cover": s["split"],
                "exposed_chars": s["exposed_chars"],
                "over_masked_chars": s["over_chars"],
            }
        payload = {
            "gold_path": os.path.abspath(args.gold),
            "pred_path": os.path.abspath(args.pred),
            "pred_format": fmt,
            "join": {"key": join_key, "ok": len(pairs), "fail": len(join_fail),
                     "total": len(docs)},
            "em_require_label": args.em_require_label,
            "span_unit": "span [start, end) char offsets",
            "match_rule": {
                "assignment": "greedy 1:1 by overlap chars desc",
                "tie_break": "(doc_id, gold.start, pred.start) asc",
                "label_rule": "pred label == gold.corp_category or gold.opf_label",
                "primary_criterion": "partial",
            },
            "standard_criteria": {
                c: {"tp": std[c]["tp"], "fn": G - std[c]["tp"],
                    "fp": P - std[c]["tp"],
                    "recall": {"num": std[c]["tp"], "den": G},
                    "precision": {"num": std[c]["tp"], "den": P},
                    "f1": round(fbeta(ratio(std[c]["tp"], P),
                                      ratio(std[c]["tp"], G), 1), 6),
                    "f2": round(fbeta(ratio(std[c]["tp"], P),
                                      ratio(std[c]["tp"], G), 2), 6)}
                for c in CRITERIA},
            "overall": blk(tot),
            "by_corp_category": {c: blk(by_cat[c]) for c in by_cat},
            "by_opf_label": {c: blk(by_opf[c]) for c in by_opf},
            "by_difficulty_case": {c: blk(by_case[c]) for c in by_case},
            "by_length_tier": {c: blk(by_tier[c]) for c in by_tier},
            "length_tier_bounds": {"q1": q1, "q2": q2},
            "doc_level": {"total": nd, "fully_masked": docs_clean,
                          "with_miss": docs_missed,
                          "miss_by_top_sensitive_category": cnt},
            "unique_id_4": {"recall": {"num": ut, "den": ug}, "fn": ug - ut,
                            "gold_coverage": {"num": uh, "den": ug},
                            "uncovered": ug - uh,
                            "allowance": UNIQUE_ID_ALLOWANCE,
                            "meets": (ug - ut) <= UNIQUE_ID_ALLOWANCE},
            "over_masked_by_category": over_by_cat,
            "unattributed_pred_spans": unattrib_pred,
            "partial_samples": partial_samples[:args.partial_sample],
            "accuracy_appendix": {"correct_chars": correct,
                                  "total_chars": tot_chars,
                                  "pii_chars": gold_chars_all},
        }
        try:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
                f.write("\n")
        except OSError as e:
            die("JSON 을 쓸 수 없습니다: %s (%s)" % (args.out_json, e))
        print("")
        print("JSON 저장 : %s" % args.out_json)
    print("")


if __name__ == "__main__":
    main()
