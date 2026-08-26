"""로짓 캐시에서 argmax 와 제약 Viterbi 두 경로를 디코딩해 비교하고,
축별 대표 케이스를 전수 선별해 발표용 좌우 비교 JSONL 을 만든다.

추론하지 않는다. cache_logits.py 가 만든 캐시만 읽는다.
난수를 쓰지 않는다. 표본 추정을 하지 않는다 — 전부 전수 집계다.
지표는 type(겹침, 라벨 무관) 하나이고 재현율이 주지표, F1 이 보조지표다.
정밀도·F2 는 내지 않는다. strict/partial 은 산출하지 않는다.

    python3 decode_compare.py --which gold --out results/showcase.jsonl
    python3 decode_compare.py --which probe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import viterbi
from postproc import group_spans, split_tag   # argmax 기준선용. 수정하지 않는다.

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "data", "ss_pii_testset_ko_v1.json")
PROBE = os.path.join(HERE, "data", "probe_partial_utterance_corp.jsonl")
RULE = os.path.join(HERE, "results", "rule_predictions.jsonl")
CACHE = os.path.expanduser("~/.opf/cache")
CKPT = os.path.expanduser("~/.opf/privacy_filter")

CORP_ITEMS = ["주소", "주민등록번호", "국문 성명", "이메일 주소", "연락처",
              "영문 성명", "카드번호", "외국인등록번호", "운전면허번호",
              "여권번호", "계좌번호"]
RULE_ITEMS = ("주민등록번호", "외국인등록번호", "여권번호", "운전면허번호")
FORM_CUES = ("성명", "직책", "담당자", "이름")
SYMBOLS = ("$", "₩", "(", "「", "《")
HONORIFICS = ("님", "씨", "수석", "팀장", "과장", "차장", "부장", "PB")
NAME_ITEMS = ("국문 성명", "영문 성명")
TOP_N = 5

SEP = re.compile(r"[\s\-–—_.·,/\\()\[\]{}:;'\"]+")
CHUNK = re.compile(r"[0-9]+|[A-Za-z]+|[가-힣]+")


def die(msg):
    print("오류: %s" % msg, file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------- 입력

def load_cache(which):
    npz = os.path.join(CACHE, "logits_%s.npz" % which)
    meta = os.path.join(CACHE, "meta_%s.jsonl" % which)
    for p in (npz, meta):
        if not os.path.isfile(p):
            die("캐시가 없습니다: %s (cache_logits.py 를 먼저 돌리십시오)" % p)
    z = np.load(npz)
    rows = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["doc_id"], r["text"],
                         [tuple(o) for o in r["offsets"]]))
    return z, rows


def load_gold_spans():
    with open(GOLD, encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    return {d["id"]: [(s["start"], s["end"], s["corp_category"], s["opf_label"])
                      for s in d["spans"]] for d in docs}


def load_probe_spans():
    out = {}
    with open(PROBE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            item = r["info"]["corp_category"]
            sp = []
            for key, ranges in r["spans"].items():
                for a, b in ranges:
                    sp.append((int(a), int(b), item, ""))
            out[r["info"]["id"]] = sorted(sp)
    return out


def load_rule_by_text():
    if not os.path.isfile(RULE):
        return {}
    by = {}
    with open(RULE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by[r["text"]] = [(int(p["start"]), int(p["end"]),
                              p.get("label") or p.get("corp_category") or "")
                             for p in (r.get("predictions") or [])]
    return by


# ---------------------------------------------------------------- 공통

def overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


MIN_OVERLAP = 3


def _hits(g, pred, min_ov=MIN_OVERLAP):
    """정답 g 를 3자 이상 덮는 예측 인덱스들. 정답이 3자 미만이면 전건 피복 요구."""
    n = g[1] - g[0]
    out = []
    for pi, p in enumerate(pred):
        if not overlap(g[0], g[1], p[0], p[1]):
            continue
        ov = min(g[1], p[1]) - max(g[0], p[0])
        if (ov >= n) if n < min_ov else (ov >= min_ov):
            out.append(pi)
    return out


def match_many(gold, pred, min_ov=MIN_OVERLAP):
    """다대일 허용 3자 기준 매칭.

    하나의 예측이 여러 정답을 3자 이상씩 덮으면 그 정답들을 전부 TP 로 인정한다.
    마스킹 결과가 실제로 전부 덮였으므로 실패로 셀 이유가 없다.
    라벨은 보지 않는다. 난수를 쓰지 않는다.

    반환: (matched_gold_idx set, matched_pred_idx set)
    """
    gm, pm = set(), set()
    for gi, g in enumerate(gold):
        h = _hits(g, pred, min_ov)
        if h:
            gm.add(gi)
            pm.update(h)
    return gm, pm


def greedy_match(gold, pred, min_ov=MIN_OVERLAP):
    """축 분류용 — 정답별 대표 예측 1개(겹침 최대)를 고른다.

    지표 산출에는 쓰지 않는다. F5_MISLABEL / F6_OVERRUN 처럼
    '어느 예측과 비교할지' 가 필요한 축에서만 쓴다.
    동점은 (gold.start, pred.start) 오름차순으로 결정적으로 정한다.
    """
    pair = {}
    for gi, g in enumerate(gold):
        best = None
        for pi in _hits(g, pred, min_ov):
            ov = min(g[1], pred[pi][1]) - max(g[0], pred[pi][0])
            key = (-ov, pred[pi][0], pi)
            if best is None or key < best[0]:
                best = (key, pi)
        if best is not None:
            pair[gi] = best[1]
    return pair


def rf1(tp, fn, fp):
    """재현율(주지표) 과 F1(보조지표) 만. 정밀도·F2 는 내지 않는다."""
    r = tp / (tp + fn) if (tp + fn) else 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return r, f1


def cov_set(spans):
    s = set()
    for a, b, *_ in spans:
        s.update(range(a, b))
    return s


def to_ranges(idxs):
    if not idxs:
        return []
    xs = sorted(idxs)
    out, st, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append([st, prev + 1])
        st = prev = x
    out.append([st, prev + 1])
    return out


def mask_stars(text, spans):
    ch = list(text)
    for a, b, *_ in spans:
        for i in range(max(0, a), min(len(ch), b)):
            ch[i] = "*"
    return "".join(ch)


def identifiable(text, gs, ge, covered):
    """F 항 — 안 덮인 나머지에서 구분자를 뗀 최대 덩어리가 원값 길이의 절반 이상."""
    left = "".join(text[i] for i in range(gs, ge) if i not in covered)
    left = SEP.sub("", left)
    if not left:
        return False, 0
    mx = max((len(m.group()) for m in CHUNK.finditer(left)), default=0)
    return mx * 2 >= (ge - gs), mx


# ---------------------------------------------------------------- 디코딩

def _span_openers(tags, offs):
    """group_spans 가 내는 스팬과 **같은 순서**로, 각 스팬을 연 토큰의
    전이 정보 (이전태그, 이전카테고리, 현재태그, 현재카테고리, 첫토큰여부) 를 낸다.

    group_spans 의 분기를 그대로 따라 걷기만 한다. 판정은 하지 않는다.
    """
    out, cur = [], None
    prev_t = prev_c = None
    first = True

    def close():
        nonlocal cur
        if cur is not None:
            out.append(cur[1])
        cur = None

    for tag, (s, e) in zip(tags, offs):
        if s == e:                              # group_spans 와 동일하게 건너뛴다
            continue
        pre, lab = split_tag(tag)
        t = None if pre == "O" else pre
        c = None if pre == "O" else lab
        info = (prev_t, prev_c, t, c, first)
        if pre == "O":
            close()
        elif pre == "S":
            close()
            out.append(info)
        elif pre == "B":
            close()
            cur = (lab, info)
        else:                                   # I / E — 고아면 여기서 스팬이 열린다
            if cur is None or cur[0] != lab:
                close()
                cur = (lab, info)
            if pre == "E":
                close()
        prev_t, prev_c = t, c
        first = False
    close()
    return out


def illegal_opened(tags, offs, n_span):
    """argmax 스팬 중 '금지 전이로 열린' 것의 인덱스 집합.

    판정 술어는 viterbi.is_valid_transition 을 그대로 쓴다 — 새 규칙을 만들지 않는다.
    시퀀스 첫 토큰 규칙도 count_illegal_transitions 와 같다(B-/S-/O 만 허용).
    """
    infos = _span_openers(tags, offs)
    if len(infos) != n_span:
        die("스팬 개수 불일치 %d != %d — 금지 전이 귀속을 중단합니다"
            % (len(infos), n_span))
    ill = set()
    for si, (pt, pc, t, c, first) in enumerate(infos):
        bad = (t not in ("B", "S")) if first else \
              (not viterbi.is_valid_transition(pt, pc, t, c))
        if bad:
            ill.add(si)
    return ill


def decode_all(z, rows, i2l, V):
    """캐시에서 argmax / Viterbi 두 경로를 낸다. 재추론하지 않는다."""
    out = {}
    arg_ill = {}
    t_arg = t_vit = 0.0
    bad_arg = bad_vit = 0
    for did, text, offs in rows:
        lg = z[did].astype(np.float32)

        t0 = time.time()
        ids_a = lg.argmax(-1).tolist()
        tags_a = [i2l[str(i)] for i in ids_a]
        sp_a = [(s["start"], s["end"], s["label"])
                for s in group_spans(tags_a, offs)]
        t_arg += time.time() - t0
        arg_ill[did] = illegal_opened(tags_a, offs, len(sp_a))

        t0 = time.time()
        ids_v = V.decode(lg)
        sp_v = [(s["start"], s["end"], s["label"])
                for s in viterbi.spans_from_bioes(ids_v, offs, i2l)]
        t_vit += time.time() - t0

        bad_arg += viterbi.count_illegal_transitions(ids_a, i2l)
        bad_vit += viterbi.count_illegal_transitions(ids_v, i2l)
        out[did] = (text, offs, sp_a, sp_v)
    return out, (t_arg, t_vit), (bad_arg, bad_vit), arg_ill


def summarize(dec, gold_by, which_idx):
    """3자 기준 지표 + 상태 3분류 + 부록 측정.

    지표(재현율·F1)는 같은 분자 TP 를 쓴다.
    3분류(완전 피복/부분 노출/완전 미탐)는 지표가 아니라 상태 분포다.
    """
    tp = fn = fp = 0
    per = defaultdict(lambda: [0, 0, 0])       # [tp, fn, fp]
    full = part = miss = 0
    part_chars = 0
    per_cov = defaultdict(lambda: [0, 0, 0])   # [full, part, miss]
    n_span = n_tok1 = 0
    tp_ident = 0                               # 부록: TP 인데 잔여 식별 가능
    tp_ident_by = Counter()
    dropped_1to3 = 0                           # 겹침 1자 기준 TP -> 3자에서 탈락

    for did, (text, offs, sp_a, sp_v) in dec.items():
        pred = (sp_a, sp_v)[which_idx]
        gold = gold_by.get(did, [])
        n_span += len(pred)
        n_tok1 += sum(1 for a, b, _ in pred if b - a == 1)

        m, pm = match_many(gold, pred)
        m1, _ = match_many(gold, pred, min_ov=1)
        dropped_1to3 += len(m1 - m)

        tp += len(m)
        fn += len(gold) - len(m)
        fp += len(pred) - len(pm)      # 어떤 정답과도 3자 이상 겹치지 않는 예측
        for gi, g in enumerate(gold):
            per[g[2]][0 if gi in m else 1] += 1
        for pi, p in enumerate(pred):
            if pi in pm:
                continue
            owner = None
            for g in gold:
                if overlap(g[0], g[1], p[0], p[1]):
                    owner = g[2]
                    break
            per[owner or "(순수오탐)"][2] += 1

        pc = cov_set(pred)
        for gi, (gs, ge, item, _) in enumerate(gold):
            n = ge - gs
            c = len(pc & set(range(gs, ge)))
            if c == n:
                full += 1
                per_cov[item][0] += 1
            elif c == 0:
                miss += 1
                per_cov[item][2] += 1
            else:
                part += 1
                per_cov[item][1] += 1
                part_chars += n - c
            # 부록 — TP 로 잡혔는데 안 덮인 나머지로 여전히 식별 가능한가
            if gi in m and c < n:
                ok, _ = identifiable(text, gs, ge, pc)
                if ok:
                    tp_ident += 1
                    tp_ident_by[item] += 1

    return {"tp": tp, "fn": fn, "fp": fp, "per": per,
            "full": full, "part": part, "miss": miss, "part_chars": part_chars,
            "per_cov": per_cov, "n_span": n_span, "n_tok1": n_tok1,
            "tp_ident": tp_ident, "tp_ident_by": tp_ident_by,
            "dropped_1to3": dropped_1to3}


# ---------------------------------------------------------------- 축 분류 (D)

def classify(dec, gold_by, rule_by_text, arg_ill=None):
    """기준은 전부 Viterbi 결과다. argmax 는 V1_DECODE / V2_REGRESS 비교에만 쓴다."""
    grows, prows, drows = [], [], []
    mis_pairs = Counter()

    for did, (text, offs, sp_a, sp_v) in dec.items():
        gold = gold_by.get(did, [])
        pc_v, pc_a = cov_set(sp_v), cov_set(sp_a)
        m, _ = match_many(gold, sp_v)          # TP 판정 (다대일)
        m_arg, _ = match_many(gold, sp_a)
        rep = greedy_match(gold, sp_v)         # 대표 예측 (F5/F6 용)
        rule = rule_by_text.get(text, [])
        rc = cov_set(rule)
        ill = (arg_ill or {}).get(did, set())

        by_item = defaultdict(list)
        all_tp = bool(gold) and len(m) == len(gold)   # 정답 스팬 전건 TP

        for gi, (gs, ge, item, glab) in enumerate(gold):
            span = set(range(gs, ge))
            cv = len(pc_v & span)
            ca = len(pc_a & span)
            n = ge - gs
            verdict = "완전미탐" if cv == 0 else ("완전탐지" if cv == n else "부분노출")
            verdict_arg = ("완전미탐" if ca == 0
                           else ("완전탐지" if ca == n else "부분노출"))
            by_item[item].append(verdict)
            # 정답 스팬과 실제로 겹친 예측 스팬들 (겹침 1자 이상, 라벨 무관)
            hit_a = [pj for pj, q in enumerate(sp_a)
                     if overlap(gs, ge, q[0], q[1])]
            hit_v = [pj for pj, q in enumerate(sp_v)
                     if overlap(gs, ge, q[0], q[1])]
            pi = rep.get(gi)
            tags = []

            # F1_ORPHAN — 부분 노출 전체가 모집단
            if verdict == "부분노출":
                tags.append("F1_ORPHAN")
            # F2_FORM
            if verdict == "완전미탐" and item in NAME_ITEMS and \
                    any(c in text[max(0, gs - 15):gs] for c in FORM_CUES):
                tags.append("F2_FORM")
            # F3_NARR
            if verdict == "완전미탐" and item == "주소" and \
                    "주소" not in text[max(0, gs - 15):gs]:
                tags.append("F3_NARR")
            # F5_MISLABEL
            if pi is not None and sp_v[pi][0] == gs and sp_v[pi][1] == ge \
                    and sp_v[pi][2] != glab:
                tags.append("F5_MISLABEL")
                mis_pairs[(glab, sp_v[pi][2])] += 1
            # F6_OVERRUN
            n_over = 0
            if pi is not None:
                ps, pe, _ = sp_v[pi]
                over = set(range(ps, gs)) | set(range(ge, pe))
                n_over = len(over)
                otxt = text[ps:gs] + text[ge:pe]
                if over and any(h in otxt for h in HONORIFICS):
                    tags.append("F6_OVERRUN")
            # H1_RULE — 고유식별 4종 중 OPF 예측과 3자 이상 겹치지 못한 것
            if item in RULE_ITEMS and gi not in m:
                tags.append("H1_RULE")
            # H2_OPF_ONLY — 규칙 레이어가 예측을 내지 못하는 항목(4종 이외) 중
            #               Viterbi OPF 가 완전 피복한 것. 규칙만으로는 못 잡는다.
            if item not in RULE_ITEMS and cv == n:
                tags.append("H2_OPF_ONLY")
            # V1_DECODE — argmax 에서 TP 가 아니었는데 Viterbi 에서 TP 가 된 정답 스팬
            if gi in m and gi not in m_arg:
                tags.append("V1_DECODE")
            # V2_REGRESS — argmax 가 최소 1자는 덮었으나 Viterbi 에서 완전미탐이 된 스팬
            if verdict_arg in ("완전탐지", "부분노출") and verdict == "완전미탐":
                tags.append("V2_REGRESS")

            ok_id, mx = (identifiable(text, gs, ge, pc_v)
                         if 0 < cv < n else (False, 0))
            grows.append({
                "doc_id": did, "start": gs, "end": ge, "item": item,
                "gold_label": glab, "verdict": verdict, "n_gold": n,
                "n_cov": cv, "n_left": n - cv, "n_cov_arg": ca,
                "n_left_arg": n - ca, "pred_idx": pi, "n_over": n_over,
                "identifiable": ok_id, "max_chunk": mx,
                "verdict_arg": verdict_arg,
                "n_hit_arg": len(hit_a), "n_hit_vit": len(hit_v),
                "n_hit_arg_ill": sum(1 for pj in hit_a if pj in ill),
                "tp_vit": gi in m, "tp_arg": gi in m_arg,
                "rule_exact": any(rs == gs and re_ == ge for rs, re_, _ in rule),
                "tags": tags})

        # F7_INCONSIST — 같은 항목이 2개 이상인데 일부만 완전 피복, 일부는 완전미탐
        for item, vs in by_item.items():
            if len(vs) >= 2 and any(v == "완전탐지" for v in vs) \
                    and any(v == "완전미탐" for v in vs):
                for r in grows:
                    if r["doc_id"] == did and r["item"] == item:
                        r["tags"].append("F7_INCONSIST")
                        r["n_miss_item"] = sum(1 for v in vs if v == "완전미탐")
                        r["n_item"] = len(vs)

        # S1_STRONG — 정답 스팬이 전건 완전 피복된 문서
        nr = [g for g in gold if g[2] not in RULE_ITEMS]          # 규칙 불가 항목
        nr_full = sum(1 for gs, ge, it, _ in nr
                      if set(range(gs, ge)) <= pc_v)
        drows.append({"doc_id": did, "strong": all_tp, "n_span": len(gold),
                      "n_norule": len(nr), "n_norule_full": nr_full,
                      "text": text, "gold": gold, "pred": sp_v,
                      "pred_arg": sp_a, "rule": rule})

        # D_UNLABELED — 순수오탐이면서 private_date
        _, pm_v = match_many(gold, sp_v)
        for pi, p in enumerate(sp_v):
            if pi in pm_v:
                continue
            if any(overlap(g[0], g[1], p[0], p[1]) for g in gold):
                continue
            tags = ["D_UNLABELED"] if p[2] == "private_date" else []
            prows.append({"doc_id": did, "start": p[0], "end": p[1],
                          "label": p[2], "length": p[1] - p[0], "tags": tags})

    return grows, prows, drows, mis_pairs


V_ORDER = ("완전탐지", "부분노출", "완전미탐")
V_NAME = {"완전탐지": "완전 피복", "부분노출": "부분 노출", "완전미탐": "완전 미탐"}


def v2_report(grows):
    """R1 상태 전이 행렬 + V2_REGRESS 모집단·원인 진단. 전수 집계, 표본 아님."""
    # ---- R1 3x3 교차표 -------------------------------------------------
    cx = {a: {b: 0 for b in V_ORDER} for a in V_ORDER}
    for r in grows:
        cx[r["verdict_arg"]][r["verdict"]] += 1
    rowsum = {a: sum(cx[a].values()) for a in V_ORDER}
    colsum = {b: sum(cx[a][b] for a in V_ORDER) for b in V_ORDER}
    total = sum(rowsum.values())

    print("[4-1] 상태 전이 행렬 — argmax 상태 x Viterbi 상태 (정답 스팬 전수)")
    print("      행 = argmax, 열 = Viterbi. 상태는 문자 피복 기준이다(지표 아님).")
    print()
    print("      %-14s %12s %12s %12s %12s"
          % ("argmax \\ Viterbi", V_NAME[V_ORDER[0]], V_NAME[V_ORDER[1]],
             V_NAME[V_ORDER[2]], "행 합계"))
    print("      " + "-" * 66)
    for a in V_ORDER:
        print("      %-14s %12d %12d %12d %12d"
              % (V_NAME[a], cx[a][V_ORDER[0]], cx[a][V_ORDER[1]],
                 cx[a][V_ORDER[2]], rowsum[a]))
    print("      " + "-" * 66)
    print("      %-14s %12d %12d %12d %12d"
          % ("열 합계", colsum[V_ORDER[0]], colsum[V_ORDER[1]],
             colsum[V_ORDER[2]], total))
    print()
    print("      총합 검증 : %d (기대 %d) %s"
          % (total, len(grows), "일치" if total == len(grows) else "불일치"))
    exp_row = (25564, 710, 2146)
    exp_col = (25859, 214, 2347)
    got_row = tuple(rowsum[a] for a in V_ORDER)
    got_col = tuple(colsum[b] for b in V_ORDER)
    print("      행 합계   : %s (기대 %s) %s"
          % (got_row, exp_row, "일치" if got_row == exp_row else "불일치"))
    print("      열 합계   : %s (기대 %s) %s"
          % (got_col, exp_col, "일치" if got_col == exp_col else "불일치"))
    if total != 28420 or got_row != exp_row or got_col != exp_col:
        die("상태 전이 행렬이 기대값과 어긋납니다 — 집계를 중단합니다")
    print()

    # ---- R2 모집단 정의 -------------------------------------------------
    v2 = [r for r in grows if "V2_REGRESS" in r["tags"]]
    inflow = cx["완전탐지"]["완전미탐"] + cx["부분노출"]["완전미탐"]
    outflow = cx["완전미탐"]["완전탐지"] + cx["완전미탐"]["부분노출"]
    net = inflow - outflow

    print("[4-2] V2_REGRESS 모집단")
    print("      정의 : argmax 가 (완전피복 또는 부분노출) 이면서 Viterbi 가 완전미탐")
    print("      유입(V2_REGRESS) %d건 = 완전피복->완전미탐 %d + 부분노출->완전미탐 %d"
          % (inflow, cx["완전탐지"]["완전미탐"], cx["부분노출"]["완전미탐"]))
    print("      태그 부여 건수 %d건 — 교차표 두 칸 합과 %s"
          % (len(v2), "일치" if len(v2) == inflow else "불일치"))
    print("      유출(반대 방향) %d건 = 완전미탐->완전피복 %d + 완전미탐->부분노출 %d"
          % (outflow, cx["완전미탐"]["완전탐지"], cx["완전미탐"]["부분노출"]))
    print("      유입 - 유출 = %d - %d = %+d  (완전미탐 순증 기대 +201) %s"
          % (inflow, outflow, net, "일치" if net == 201 else "불일치"))
    if len(v2) != inflow:
        die("V2_REGRESS 태그 건수가 교차표와 어긋납니다 — 중단합니다")
    print()

    # ---- R3 원인 진단 ---------------------------------------------------
    n = len(v2)
    print("[4-3] 원인 진단")
    print("  (a) argmax 가 덮은 문자 수 분포 (모집단 %d건)" % n)
    bins = [("1자", lambda c: c == 1), ("2자", lambda c: c == 2),
            ("3~5자", lambda c: 3 <= c <= 5), ("6자 이상", lambda c: c >= 6)]
    print("      %-10s %8s %14s %10s" % ("구간", "건수", "분수", "백분율"))
    print("      " + "-" * 46)
    chk = 0
    for name, f in bins:
        k = sum(1 for r in v2 if f(r["n_cov_arg"]))
        chk += k
        print("      %-10s %8d %14s %9.2f%%"
              % (name, k, "%d/%d" % (k, n), 100.0 * k / n if n else 0.0))
    print("      " + "-" * 46)
    print("      %-10s %8d %14s %9.2f%%"
          % ("합계", chk, "%d/%d" % (chk, n), 100.0 * chk / n if n else 0.0))
    covs = sorted(r["n_cov_arg"] for r in v2)
    if covs:
        print("      최소 %d자 / 중앙값 %d자 / 최대 %d자 / 평균 %.2f자"
              % (covs[0], covs[len(covs) // 2], covs[-1],
                 sum(covs) / float(len(covs))))
    print()

    print("  (b) Viterbi 예측 스팬과의 겹침 이분")
    z0 = sum(1 for r in v2 if r["n_hit_vit"] == 0)
    z1 = n - z0
    print("      겹친 Viterbi 예측 스팬 0개            : %d건 (%d/%d)" % (z0, z0, n))
    print("      겹쳤으나 3자 미만이라 미탐 처리       : %d건 (%d/%d)" % (z1, z1, n))
    print("      주: 이 코드에서 '완전미탐' 의 정의는 문자 피복 0 (cv == 0) 이다.")
    print("          따라서 1자라도 겹치면 '부분노출' 로 가고 완전미탐이 될 수 없어")
    print("          두 번째 칸은 정의상 항상 0 이다. 3자 규칙과는 무관한 축이다.")
    ntp = sum(1 for r in v2 if not r["tp_vit"])
    print("      참고(3자 매칭 기준) : V2_REGRESS 중 Viterbi 비TP %d/%d" % (ntp, n))
    print()

    print("  (c) 해당 스팬을 덮었던 argmax 예측 스팬 중 BIOES 금지 전이로 열린 것")
    print("      판정 술어는 viterbi.is_valid_transition 재사용 (신규 규칙 없음).")
    tot_hit = sum(r["n_hit_arg"] for r in v2)
    tot_ill = sum(r["n_hit_arg_ill"] for r in v2)
    any_ill = sum(1 for r in v2 if r["n_hit_arg_ill"] > 0)
    all_ill = sum(1 for r in v2 if r["n_hit_arg"] > 0
                  and r["n_hit_arg_ill"] == r["n_hit_arg"])
    print("      겹친 argmax 예측 스팬 총계             : %d개" % tot_hit)
    print("      그중 금지 전이로 열린 것               : %d개 (%d/%d)"
          % (tot_ill, tot_ill, tot_hit))
    print("      금지 전이 스팬을 1개 이상 가진 정답 스팬: %d건 (%d/%d)"
          % (any_ill, any_ill, n))
    print("      겹친 argmax 스팬이 전부 금지 전이인 정답 스팬: %d건 (%d/%d)"
          % (all_ill, all_ill, n))
    print()

    print("  (d) 항목별 V2_REGRESS 건수 (내림차순)")
    by_item = Counter(r["item"] for r in v2)
    print("      %-16s %8s %14s" % ("항목", "건수", "분수"))
    print("      " + "-" * 42)
    for it, k in sorted(by_item.items(), key=lambda kv: (-kv[1], kv[0])):
        print("      %-16s %8d %14s" % (it, k, "%d/%d" % (k, n)))
    print("      " + "-" * 42)
    print("      %-16s %8d" % ("합계", sum(by_item.values())))
    print()
    return v2


V2_BINS = (("B1", "1자", 1), ("B2", "2자", 1),
           ("B3", "3~5자", 1), ("B4", "6자 이상", 2))
V2_EXPECT = {"B1": 56, "B2": 44, "B3": 61, "B4": 46}


def v2_bin(c):
    """argmax 피복 문자 수 -> 구간 이름."""
    if c <= 1:
        return "B1"
    if c == 2:
        return "B2"
    if c <= 5:
        return "B3"
    return "B4"


def pick_v2(v2, n=TOP_N):
    """V2_REGRESS 대표 케이스를 피복 문자 수 구간별로 층화 선별한다.

    전수 모집단을 B1(1자)/B2(2자)/B3(3~5자)/B4(6자 이상) 로 나누고
    할당량 1/1/1/2 로 채운다. 난수를 쓰지 않는다 — 정렬은 전부 결정적이다.

    구간 내 정렬 : 정답 스팬 길이 내림차순 -> 피복 문자 수 내림차순
                   -> (doc_id, gold.start) 오름차순
    제약         : 문서당 1건(완화하지 않음), 항목당 2건
    항목 상한 때문에 구간을 못 채우면 R5 순서로만 완화하고 그 사실을 출력한다.
    """
    pools = defaultdict(list)
    for r in v2:
        pools[v2_bin(r["n_cov_arg"])].append(r)
    for k in pools:
        pools[k].sort(key=lambda r: (-r["n_gold"], -r["n_cov_arg"],
                                     r["doc_id"], r["start"]))

    print("[4-4] V2_REGRESS 대표 케이스 — 피복 문자 수 구간별 층화 선별")
    print("      %-6s %-10s %8s %8s" % ("구간", "정의", "모집단", "할당"))
    print("      " + "-" * 38)
    tot = 0
    bad = []
    for key, name, quota in V2_BINS:
        k = len(pools[key])
        tot += k
        if k != V2_EXPECT[key]:
            bad.append((key, k, V2_EXPECT[key]))
        print("      %-6s %-10s %8d %8d" % (key, name, k, quota))
    print("      " + "-" * 38)
    print("      %-6s %-10s %8d %8d" % ("합계", "", tot, n))
    if bad or tot != len(v2):
        die("구간 건수가 기대값과 어긋납니다 %s (합 %d / 모집단 %d)"
            % (bad, tot, len(v2)))
    print("      구간 건수 검증 : %s (기대 %s) 일치 / 합 %d"
          % (tuple(len(pools[k]) for k, _, _ in V2_BINS),
             tuple(V2_EXPECT[k] for k, _, _ in V2_BINS), tot))
    print()

    seen_doc, cnt_item, picked = set(), Counter(), []
    relax = []

    def take(key, cap):
        """구간 key 에서 항목 상한 cap 으로 1건 뽑는다. 성공하면 True."""
        for r in pools[key]:
            if r in picked or r["doc_id"] in seen_doc:
                continue
            if cnt_item[r["item"]] >= cap:
                continue
            seen_doc.add(r["doc_id"])
            cnt_item[r["item"]] += 1
            picked.append(r)
            r["_bin"] = key
            return True
        return False

    order = [k for k, _, _ in V2_BINS]
    for key, name, quota in V2_BINS:
        for _ in range(quota):
            if take(key, 2):
                continue
            # R5-1) 이 구간에 한해 항목 상한 2 -> 3 완화
            if take(key, 3):
                relax.append("%s 구간 — 항목 상한 2 -> 3 완화" % key)
                continue
            # R5-2) 인접 구간에서 대체 (B4 는 B3, 그 외는 번호가 큰 쪽)
            i = order.index(key)
            alt = order[i - 1] if key == "B4" else order[i + 1]
            done = False
            for cap in (2, 3):
                if take(alt, cap):
                    relax.append("%s 구간 미충족, %s 에서 대체%s"
                                 % (key, alt, " (항목 상한 3 완화)"
                                    if cap == 3 else ""))
                    done = True
                    break
            if not done:
                relax.append("%s 구간 미충족, %s 대체도 실패" % (key, alt))

    if relax:
        print("      R5 대체 규칙 적용 내역")
        for s in relax:
            print("        - %s" % s)
    else:
        print("      R5 대체 규칙 : 적용 없음 (할당량 그대로 충족)")
    if len(picked) != n:
        die("V2_REGRESS 대표 케이스가 %d건입니다 — %d건이 아닙니다"
            % (len(picked), n))
    print()
    print("      %-14s %-12s %8s %8s %6s"
          % ("doc_id", "항목", "arg피복", "정답길이", "구간"))
    print("      " + "-" * 54)
    for r in picked:
        print("      %-14s %-12s %8d %8d %6s"
              % (r["doc_id"], r["item"], r["n_cov_arg"], r["n_gold"],
                 r["_bin"]))
    print()
    return picked


def pick(rows, tag, keyfn, n=TOP_N, one_per_doc=False):
    """결정적 선별. 난수를 쓰지 않는다.

    one_per_doc=True 면 문서당 1건만 남긴다 — 축의 취지가 문서 단위일 때 쓴다.
    """
    sel = [r for r in rows if tag in r["tags"]]
    sel.sort(key=lambda r: (keyfn(r), r["doc_id"], r["start"]))
    if one_per_doc:
        seen, out = set(), []
        for r in sel:
            if r["doc_id"] in seen:
                continue
            seen.add(r["doc_id"])
            out.append(r)
        sel = out
    return sel[:n]


# ---------------------------------------------------------------- 산출물 (E)

def build_record(tag, item, note, doc, extra=False):
    text = doc["text"]
    gold, pred, parg, rule = doc["gold"], doc["pred"], doc["pred_arg"], doc["rule"]
    gc, pc = cov_set(gold), cov_set(pred)
    rec = {
        "doc_id": doc["doc_id"], "tag": tag, "item": item, "note": note,
        "source_text": text,
        "masked_viterbi": mask_stars(text, pred),
        "masked_argmax": mask_stars(text, parg),
        "gold_spans": [[g[0], g[1], g[2]] for g in gold],
        "pred_spans": [[p[0], p[1], p[2]] for p in pred],
        "leaked_spans": to_ranges(gc - pc),
        "overrun_spans": to_ranges(pc - gc),
    }
    if extra:
        rec["masked_opf"] = rec["masked_viterbi"]
        rec["masked_rule"] = mask_stars(text, rule)
        rec["masked_hybrid"] = mask_stars(text, list(pred) + list(rule))
        rec["rule_spans"] = [[r[0], r[1], r[2]] for r in rule]
    return rec


def emit(dec, gold_by, rule_by_text, out_path, arg_ill=None):
    grows, prows, drows, mis = classify(dec, gold_by, rule_by_text, arg_ill)
    by_doc = {d["doc_id"]: d for d in drows}

    gt = Counter(t for r in grows for t in r["tags"])
    pt = Counter(t for r in prows for t in r["tags"])
    n_strong = sum(1 for d in drows if d["strong"])

    print("[4] 축별 모집단 (3,000건 전수 집계, 표본 아님)")
    print("    미탐 축(정답 스팬 %d) 과 과탐 축(순수오탐 %d) 은 모집단이 다르다."
          % (len(grows), len(prows)))
    print()
    DESC = {
        "F1_ORPHAN": "부분 노출 전체",
        "F2_FORM": "성명 완전미탐 & 앞 15자 양식 단서",
        "F3_NARR": "주소 완전미탐 & 앞 15자에 '주소' 없음",
        "F5_MISLABEL": "오프셋 완전일치인데 라벨 불일치",
        "F6_OVERRUN": "경계 과확장 & 초과분에 경칭/직함",
        "F7_INCONSIST": "동일 항목 2+ 중 일부만 피복·일부 완전미탐",
        "H1_RULE": "고유식별 4종 중 OPF 예측과 3자 미달",
        "H2_OPF_ONLY": "규칙 불가 항목 중 OPF 가 완전 피복",
        "V1_DECODE": "argmax 非TP -> Viterbi TP",
        "V2_REGRESS": "argmax 1자 이상 피복 -> Viterbi 완전미탐",
    }
    print("    %-14s %-10s %8s   %s" % ("축", "모집단축", "건수", "정의"))
    print("    " + "-" * 78)
    for t in ("F1_ORPHAN", "F2_FORM", "F3_NARR", "F5_MISLABEL",
              "F6_OVERRUN", "F7_INCONSIST", "H1_RULE", "H2_OPF_ONLY",
              "V1_DECODE", "V2_REGRESS"):
        print("    %-14s %-10s %8d   %s" % (t, "정답스팬", gt.get(t, 0), DESC[t]))
    print("    %-14s %-10s %8d   %s"
          % ("D_UNLABELED", "예측스팬", pt.get("D_UNLABELED", 0),
             "순수오탐 & 예측 라벨 private_date"))
    print("    %-14s %-10s %8d   %s"
          % ("S1_STRONG", "문서", n_strong, "정답 스팬 전건 TP 인 문서"))
    n_sym = sum(1 for r in grows if "F1_ORPHAN" in r["tags"]
                and r["start"] > 0 and dec[r["doc_id"]][0][r["start"] - 1] in SYMBOLS)
    print()
    print("    F1_ORPHAN 하위 플래그 symbol=true : %d건" % n_sym)
    h1 = [r for r in grows if "H1_RULE" in r["tags"]]
    h1_part = sum(1 for r in h1 if r["verdict"] == "부분노출")
    h1_miss = sum(1 for r in h1 if r["verdict"] == "완전미탐")
    h1_full = sum(1 for r in h1 if r["verdict"] == "완전탐지")
    print("    H1_RULE 내역 : 완전피복이나 3자미달 %d + 부분노출 %d + 완전미탐 %d = %d"
          % (h1_full, h1_part, h1_miss, len(h1)))
    print()

    v2 = v2_report(grows)

    recs = []

    def add(tag, rows, keyfn, note_fn, extra=False, one_per_doc=False):
        for r in pick(rows, tag, keyfn, one_per_doc=one_per_doc):
            doc = by_doc[r["doc_id"]]
            recs.append(build_record(tag, r.get("item") or "(정본 무라벨)",
                                     note_fn(r, doc), doc, extra))

    add("F1_ORPHAN", grows, lambda r: (r["n_cov"], -r["n_left"]),
        lambda r, d: "정답 %d자 중 %d자만 덮임 — %d자 노출%s"
        % (r["n_gold"], r["n_cov"], r["n_left"],
           " (직전 문자 %r)" % d["text"][r["start"] - 1]
           if r["start"] > 0 and d["text"][r["start"] - 1] in SYMBOLS else ""))
    add("F2_FORM", grows, lambda r: -r["n_gold"],
        lambda r, d: "양식 단서 뒤 %s %d자 완전미탐" % (r["item"], r["n_gold"]))
    add("F3_NARR", grows, lambda r: -r["n_gold"],
        lambda r, d: "'주소' 단서 없는 서술문 안 주소 %d자 완전미탐" % r["n_gold"])
    add("F5_MISLABEL", grows,
        lambda r: -mis[(r["gold_label"],
                        by_doc[r["doc_id"]]["pred"][r["pred_idx"]][2])],
        lambda r, d: "오프셋 정확·라벨 %s -> %s"
        % (r["gold_label"], d["pred"][r["pred_idx"]][2]))
    add("F6_OVERRUN", grows, lambda r: -r["n_over"],
        lambda r, d: "경계 %d자 과확장 — 경칭·직함 삼킴" % r["n_over"])
    add("F7_INCONSIST", grows,
        lambda r: (-r.get("n_miss_item", 0), -r.get("n_item", 0)),
        lambda r, d: "같은 항목 %d개 중 %d개 완전미탐 — 문서 내 불일치"
        % (r.get("n_item", 0), r.get("n_miss_item", 0)), one_per_doc=True)
    add("V1_DECODE", grows, lambda r: -r["n_left_arg"],
        lambda r, d: "argmax 는 %d자 중 %d자만 덮어 TP 실패, Viterbi 가 TP (%d자 피복)"
        % (r["n_gold"], r["n_cov_arg"], r["n_cov"]))
    # V2_REGRESS — 피복 문자 수 구간별 층화 선별. 결정적이며 난수를 쓰지 않는다.
    for r in pick_v2(v2):
        recs.append(build_record(
            "V2_REGRESS", r["item"],
            "argmax 피복 %d자 / 정답 스팬 %d자" % (r["n_cov_arg"], r["n_gold"]),
            by_doc[r["doc_id"]]))

    add("D_UNLABELED", prows, lambda r: -r["length"],
        lambda r, d: "정본 무라벨 구간을 private_date 로 탐지 (%d자)" % r["length"])

    for d in sorted([x for x in drows if x["strong"]],
                    key=lambda x: (-x["n_span"], x["doc_id"]))[:TOP_N]:
        recs.append(build_record(
            "S1_STRONG", "(문서 전건)",
            "정답 스팬 %d개 전건 TP" % d["n_span"], d))

    # H1_RULE — 4종 라운드로빈
    pool = defaultdict(list)
    for r in sorted([r for r in grows if "H1_RULE" in r["tags"]],
                    key=lambda r: (r["doc_id"], r["start"])):
        pool[r["item"]].append(r)
    picked, i = [], 0
    while len(picked) < TOP_N and any(pool[k] for k in RULE_ITEMS):
        k = RULE_ITEMS[i % len(RULE_ITEMS)]
        if pool[k]:
            picked.append(pool[k].pop(0))
        i += 1
    for r in picked:
        d = by_doc[r["doc_id"]]
        recs.append(build_record(
            "H1_RULE", r["item"],
            "OPF 가 %s %d자를 완전히 덮지 못함(%s) — 규칙 레이어 경계 %s"
            % (r["item"], r["n_gold"], r["verdict"],
               "정확" if r["rule_exact"] else "불일치"),
            d, extra=True))

    # H2_OPF_ONLY — 규칙 불가 항목을 많이 가졌고 그것을 전부 완전 피복한 문서
    #               문서당 1건, 규칙불가 정답 스팬 수 내림차순
    h2_docs = [d for d in drows
               if d["n_norule"] > 0 and d["n_norule_full"] == d["n_norule"]]
    h2_docs.sort(key=lambda d: (-d["n_norule"], d["doc_id"]))
    for d in h2_docs[:TOP_N]:
        recs.append(build_record(
            "H2_OPF_ONLY", "(규칙 불가 항목)",
            "규칙 레이어가 못 잡는 항목 %d개를 OPF 가 전부 완전 피복 "
            "(규칙 예측 %d건)" % (d["n_norule"], len(d["rule"])),
            d, extra=True))

    outdir = os.path.dirname(os.path.abspath(out_path))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for r in recs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[5] 대표 케이스 (원문은 JSONL 에만)")
    print("    %-14s %-16s %-14s %s" % ("축", "doc_id", "항목", "note"))
    print("    " + "-" * 94)
    for r in recs:
        print("    %-14s %-16s %-14s %s"
              % (r["tag"], r["doc_id"], r["item"][:14], r["note"][:48]))
    print()

    bad = 0
    for r in recs:
        for k in r:
            if k.startswith("masked_") and len(r[k]) != len(r["source_text"]):
                bad += 1
    print("[6] 문자 수 보존 마스킹 검증")
    print("    레코드 %d건 / masked_* 길이 불일치 %d건" % (len(recs), bad))
    print("    출력: %s" % out_path)
    return grows


def main():
    ap = argparse.ArgumentParser(prog="decode_compare.py")
    ap.add_argument("--which", default="gold", choices=["gold", "probe"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    i2l = json.load(open(os.path.join(args.ckpt, "config.json"),
                        encoding="utf-8"))["id2label"]
    biases, cal_path = viterbi.load_biases(args.ckpt)
    V = viterbi.ConstrainedViterbi(i2l, biases)

    print("[1] BIOES 허용 전이 (캘리브레이션 %s)" % os.path.basename(cal_path))
    for k, v in biases.items():
        print("    %-38s %s" % (k, v))
    print()
    groups, rows = V.transition_table()
    M = {True: "허용", False: "금지", "same": "허용(동일만)",
         "diff": "금지(타)", "both": "허용"}
    print("    이전 \\ 다음   " + "".join("%-14s" % g for g in groups))
    print("    " + "-" * 76)
    for name, row in rows:
        print("    %-12s " % name + "".join("%-14s" % M.get(o, str(o)) for o in row))
    print()

    z, meta = load_cache(args.which)
    gold_by = load_gold_spans() if args.which == "gold" else load_probe_spans()
    rule_by = load_rule_by_text() if args.which == "gold" else {}
    print("    캐시 문서 %d / 정답 보유 %d" % (len(meta), len(gold_by)))
    print()

    dec, (t_a, t_v), (bad_a, bad_v), arg_ill = decode_all(z, meta, i2l, V)
    A = summarize(dec, gold_by, 0)
    B = summarize(dec, gold_by, 1)
    ra, f1a = rf1(A["tp"], A["fn"], A["fp"])
    rv, f1v = rf1(B["tp"], B["fn"], B["fp"])
    tot_gold = A["full"] + A["part"] + A["miss"]

    print("[2] argmax vs 제약 Viterbi — 3자 기준")
    print("    정답·예측이 3자 이상 겹치면 매칭 후보. 정답이 3자 미만이면 전량 피복 필요.")
    print("    다대일 허용 — 예측 하나가 여러 정답을 덮으면 그 정답 전부 TP.")
    print("    FP = 어떤 정답과도 3자 이상 겹치지 않는 예측. 라벨 무관. 난수 미사용.")
    print()
    print("    %-26s %14s %14s %12s" % ("항목", "argmax", "Viterbi", "차이"))
    print("    " + "-" * 70)

    def row(name, a, b, fmt="%d"):
        d = b - a
        sg = ("%+d" % d) if fmt == "%d" else ("%+.4f" % d)
        print("    %-26s %14s %14s %12s"
              % (name, fmt % a, fmt % b, sg if d else "-"))

    row("재현율", ra, rv, "%.4f")
    row("F1", f1a, f1v, "%.4f")
    print("    " + "-" * 70)
    row("TP", A["tp"], B["tp"])
    row("FN", A["fn"], B["fn"])
    row("FP", A["fp"], B["fp"])
    print("    " + "-" * 70)
    row("예측 스팬 수", A["n_span"], B["n_span"])
    row("금지 전이 발생", bad_a, bad_v)
    row("길이 1토큰 스팬", A["n_tok1"], B["n_tok1"])
    print("    %-26s %14.2f %14.2f %12s"
          % ("디코딩 시간(초)", t_a, t_v, "%+.2f" % (t_v - t_a)))
    print()
    print("    겹침 1자 기준에서는 TP 였으나 3자 기준에서 탈락한 스팬")
    print("      argmax %d건 / Viterbi %d건" % (A["dropped_1to3"], B["dropped_1to3"]))
    print()

    if bad_v != 0:
        die("Viterbi 금지 전이 %d건 — 구현이 틀렸습니다. 중단합니다." % bad_v)

    print("[2-1] 상태 3분류 (지표 아님 — 상태 분포)")
    print("    %-26s %14s %14s %12s" % ("", "argmax", "Viterbi", "차이"))
    print("    " + "-" * 70)
    row("완전 피복", A["full"], B["full"])
    row("부분 노출", A["part"], B["part"])
    row("완전 미탐", A["miss"], B["miss"])
    print("    %-26s %14d %14d %12s"
          % ("  합(=정답 스팬)", tot_gold, B["full"] + B["part"] + B["miss"], "-"))
    row("  부분 노출 문자 수", A["part_chars"], B["part_chars"])
    print()

    print("[2-2] 부록 참고치 (지표 아님)")
    print("    3자 이상 덮여 TP 로 잡혔으나 안 덮인 나머지로 여전히 식별 가능한 건수")
    print("      argmax %d건 / Viterbi %d건" % (A["tp_ident"], B["tp_ident"]))
    if B["tp_ident_by"]:
        print("      Viterbi 항목별: " + " / ".join(
            "%s %d" % (k, v) for k, v in B["tp_ident_by"].most_common()))
    print()

    print("[3] 사내 항목별 (재현율 / F1 — 3자 기준)")
    print("    %-14s %9s %9s %9s %9s %10s"
          % ("항목", "R(arg)", "R(vit)", "F1(arg)", "F1(vit)", "ΔR"))
    print("    " + "-" * 66)
    worse = []
    items = CORP_ITEMS if args.which == "gold" else sorted(
        set(g[2] for v in gold_by.values() for g in v))
    for it in items:
        r1, g1 = rf1(*A["per"][it])
        r2, g2 = rf1(*B["per"][it])
        flag = ""
        if r2 < r1 - 1e-12:
            worse.append((it, r1, r2, A["per"][it][0], B["per"][it][0]))
            flag = "  ↓"
        print("    %-14s %9.4f %9.4f %9.4f %9.4f %10s%s"
              % (it, r1, r2, g1, g2, "%+.4f" % (r2 - r1), flag))
    print()
    if worse:
        print("    Viterbi 가 재현율이 낮아진 항목 %d개 — 포장 없이 그대로:" % len(worse))
        for it, r1, r2, t1, t2 in worse:
            print("      %-14s %.4f -> %.4f (%+.4f)  TP %d -> %d"
                  % (it, r1, r2, r2 - r1, t1, t2))
    else:
        print("    Viterbi 가 재현율이 낮아진 항목: 없음")
    print()

    if args.which == "gold" and rule_by:
        print("[4] 고유식별 4종 4단 표 (범위 %d 스팬)"
              % sum(1 for v in gold_by.values() for g in v if g[2] in RULE_ITEMS))
        cols = {}
        for name in ("OPF 단독", "규칙 단독", "OPF ∪ 규칙"):
            cols[name] = [0, 0, 0, 0]      # full, part, miss, 순수오탐
        for did, (text, offs, sp_a, sp_v) in dec.items():
            rule = rule_by.get(text, [])
            gold = [g for g in gold_by.get(did, []) if g[2] in RULE_ITEMS]
            gold_all = gold_by.get(did, [])
            for name, pred in (("OPF 단독", sp_v), ("규칙 단독", rule),
                               ("OPF ∪ 규칙", list(sp_v) + list(rule))):
                pc = cov_set(pred)
                for gs, ge, item, _ in gold:
                    n = ge - gs
                    c = len(pc & set(range(gs, ge)))
                    cols[name][0 if c == n else (2 if c == 0 else 1)] += 1
                for a, b, _ in pred:
                    if not any(overlap(a, b, g[0], g[1]) for g in gold_all):
                        cols[name][3] += 1
        print("    %-16s %12s %12s %14s" % ("구분", "OPF 단독", "규칙 단독", "OPF ∪ 규칙"))
        print("    " + "-" * 58)
        for k, idx in (("완전 피복", 0), ("부분 노출", 1), ("완전 미탐", 2),
                       ("순수 오탐 스팬", 3)):
            print("    %-16s %12d %12d %14d"
                  % (k, cols["OPF 단독"][idx], cols["규칙 단독"][idx],
                     cols["OPF ∪ 규칙"][idx]))
        u_miss = cols["OPF ∪ 규칙"][2]
        print()
        print("    OPF ∪ 규칙 완전 미탐: %d건 %s"
              % (u_miss, "— 0 달성" if u_miss == 0 else "— 0 이 아닙니다"))
        print()

    if args.out:
        emit(dec, gold_by, rule_by, args.out, arg_ill)


if __name__ == "__main__":
    main()
