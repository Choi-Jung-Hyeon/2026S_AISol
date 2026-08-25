#!/usr/bin/env python3
"""OPF 예측 · 규칙 예측 · 정본을 대조해 발표용 사례를 6개 버킷으로 뽑는다.

발표에서 필요한 것은 지표가 아니라 "이 문장에서 이 값을 놓쳤다" 는 실물이다.
집계 스크립트(standalone_metrics / doc_level_miss / fp_breakdown)가 세는 것과
같은 모집단에서, 사람이 슬라이드에 그대로 붙일 수 있는 줄만 골라 출력한다.

버킷
  A  OPF 완전 미탐 중 문맥 의존 3항목(국문 성명 / 주소 / 영문 성명)
  B  OPF 완전 미탐 중 corp_group == 고유식별정보 (4항목 각 3건)
  C  과탐 - 경계 과확장 : 예측이 정답을 통째로 덮되 초과 문자가 있는 것
  D  과탐 - 순수 오탐   : 정답과 1자도 겹치지 않는 예측
  E  프로브셋에서 규칙 0건 + OPF 미탐 (P1/P2/P3 각 3건)
  F  규칙은 정확히(경계 일치) 탐지했고 OPF 는 미탐인 정답

매칭 규칙은 기존 스크립트와 같은 비대칭 축을 그대로 쓴다.
  미탐 판정(재현율 축) : 정답 스팬과 1자도 겹치는 예측이 없음
  과탐 판정(정밀도 축) : 예측 ⊆ 정답 을 만족하지 못한 예측
두 축은 모집단이 다르므로 합산하지 않는다.

조인은 text 완전 일치로만 한다. 하니스 predictions JSONL 의 example_id 는
sha256 자동 생성값이라 정본 id 와 다르기 때문이다. 조인 실패가 1건이라도
있으면 사례를 뽑지 않고 exit 2 로 중단한다.

선별에 난수를 쓰지 않는다. (doc_id, span.start) 오름차순 정렬 후 앞에서부터
필요한 건수만 취하므로 재실행 시 같은 사례가 같은 순서로 나온다.

콘솔에만 출력한다. 파일은 쓰지 않는다. 표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import sys
import unicodedata

# standalone_metrics.py / hybrid_merge.py 에서 쓰는 키 목록과 동일하게 맞춘다.
TEXT_KEYS = ("text", "input_text", "input", "content", "document")
SPAN_KEYS = ("predictions", "predicted_spans", "pred_spans", "prediction",
             "spans", "entities", "pred")
ID_KEYS = ("id", "doc_id", "example_id")

CONTEXT_CHARS = 15      # 정답값 앞뒤 문맥
MAX_LINE = 200          # 한 줄 최대 글자 수
NONE_MARK = "(없음)"

CTX_A = ("국문 성명", "주소", "영문 성명")          # 버킷 A — 문맥 의존 3항목
UNIQUE_GROUP = "고유식별정보"                        # 버킷 B — corp_group 값
UNIQUE_CATS = ("주민등록번호", "외국인등록번호", "여권번호", "운전면허번호")
PROBE_TYPES = ("P1", "P2", "P3")
FIXED_QUOTA = 3         # 버킷 B / E 는 항목별 3건 고정


def die(msg, code):
    sys.stderr.write("[dump_examples] %s\n" % msg)
    raise SystemExit(code)


def need_file(path, what):
    """입력 파일 부재 — 스택트레이스 대신 메시지 + exit 1."""
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path), 1)


# ── 출력 유틸 (standalone_metrics.dwidth/pad 와 동일 규칙) ────
def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w):
    s = str(s)
    return s + " " * max(0, w - dwidth(s))


def flat(s):
    """개행·탭이 표를 깨뜨리지 않게 한 줄로 만든다."""
    return " ".join(str(s).split())


def cut(s, w):
    """표시 폭 w 로 자른다(한글 2폭). 잘렸으면 … 를 붙인다."""
    s = flat(s)
    if dwidth(s) <= w:
        return s
    out, acc = [], 0
    for ch in s:
        cw = dwidth(ch)
        if acc + cw > w - 1:
            break
        out.append(ch)
        acc += cw
    return "".join(out) + "…"


def clip_line(s):
    """어떤 줄도 MAX_LINE 글자를 넘기지 않는다."""
    return s if len(s) <= MAX_LINE else s[:MAX_LINE - 1] + "…"


def num(n):
    return "{:,}".format(n)


def section(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ── 입력 로딩 ────────────────────────────────────────────────
def parse_spans(raw):
    """예측 스팬을 (start, end, label) 로 정규화한다.

    standalone_metrics.norm_spans / hybrid_merge.parse_spans 와 같은 규칙.
    dict 형({"항목: 값": [[s, e]]})과 list 형 모두 받는다.
    """
    out = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        for key, offs in raw.items():
            label = key.split(":", 1)[0].strip() if ":" in key else key
            for off in offs or []:
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


def load_gold(path):
    """정본 JSON — documents 배열을 그대로 돌려준다."""
    need_file(path, "정본 JSON")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        die("정본 JSON 파싱 실패: %s (%s)" % (path, e), 1)
    except OSError as e:
        die("정본 JSON 을 읽을 수 없습니다: %s (%s)" % (path, e), 1)
    docs = data.get("documents") if isinstance(data, dict) else data
    if not isinstance(docs, list) or not docs:
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % path, 1)
    return docs


def load_pred_by_text(path, what):
    """예측 JSONL — text 를 키로 스팬 목록을 담는다 (hybrid_merge 와 동일)."""
    need_file(path, what)
    by_text, n, notext, bad = {}, 0, 0, 0
    try:
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
                if not isinstance(r, dict):
                    bad += 1
                    continue
                n += 1
                text = None
                for k in TEXT_KEYS:
                    if isinstance(r.get(k), str):
                        text = r[k]
                        break
                if text is None:
                    notext += 1
                    continue
                raw = None
                for k in SPAN_KEYS:
                    if k in r:
                        raw = r[k]
                        break
                try:
                    by_text[text] = parse_spans(raw)
                except (TypeError, ValueError) as e:
                    die("%s 의 스팬 오프셋이 정수가 아닙니다: %s" % (what, e), 1)
    except OSError as e:
        die("%s 을 읽을 수 없습니다: %s (%s)" % (what, path, e), 1)
    if not n:
        die("%s 에 레코드가 없습니다: %s" % (what, path), 1)
    if bad:
        sys.stderr.write("[경고] %s JSON 파싱 실패 라인 %d 건은 건너뜀\n" % (what, bad))
    if notext:
        sys.stderr.write("[경고] %s 에 text 가 없는 레코드 %d 건은 조인 불가\n"
                         % (what, notext))
    return by_text, n


def load_probe_gold(path):
    """프로브 정본 JSONL — text / spans(dict) / info 를 doc 형태로 맞춘다.

    정본 JSON 의 documents 와 같은 모양({id, text, spans:[{start,end,value,
    corp_category,...}]})으로 정규화해 이후 로직을 공유한다.
    """
    need_file(path, "프로브 정본 JSONL")
    docs = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    sys.stderr.write("[경고] 프로브 정본 %d 행 파싱 실패 — 건너뜀\n" % ln)
                    continue
                text = r.get("text")
                if not isinstance(text, str):
                    continue
                info = r.get("info") or {}
                spans = []
                for s, e, label in parse_spans(r.get("spans")):
                    spans.append({
                        "start": s, "end": e, "value": text[s:e],
                        "corp_category": info.get("corp_category") or label,
                        "corp_group": "", "opf_label": label, "injected": False,
                    })
                spans.sort(key=lambda g: (g["start"], g["end"]))
                docs.append({
                    "id": info.get("id") or r.get("id") or "PROBE-%05d" % ln,
                    "text": text, "spans": spans,
                    "probe_type": info.get("probe_type") or "?",
                })
    except OSError as e:
        die("프로브 정본 JSONL 을 읽을 수 없습니다: %s (%s)" % (path, e), 1)
    if not docs:
        die("프로브 정본 JSONL 에 레코드가 없습니다: %s" % path, 1)
    return docs


# ── 스팬 유틸 ────────────────────────────────────────────────
def overlaps(as_, ae, bs, be):
    return as_ < be and bs < ae


def hits(spans, s, e):
    """[s, e) 와 1자라도 겹치는 스팬 목록."""
    return [t for t in spans if overlaps(t[0], t[1], s, e)]


def render_spans(text, spans, empty=NONE_MARK):
    """스팬 목록을 '표면형[라벨]' 로 표시한다."""
    if not spans:
        return empty
    return " / ".join("%s[%s]" % (flat(text[s:e]), lb or "?")
                      for s, e, lb in spans[:3]) + (" …" if len(spans) > 3 else "")


def context(text, s, e):
    """정답값 앞뒤 ±CONTEXT_CHARS 자를 「…」 로 감싼다."""
    a, b = max(0, s - CONTEXT_CHARS), min(len(text), e + CONTEXT_CHARS)
    body = flat(text[a:s]) + "《" + flat(text[s:e]) + "》" + flat(text[e:b])
    head = "…" if a > 0 else ""
    tail = "…" if b < len(text) else ""
    return "「%s%s%s」" % (head, body, tail)


# ── 조인 ─────────────────────────────────────────────────────
def join(docs, by_text, what):
    """text 완전 일치 조인. (성공 쌍, 실패 id 목록) 을 돌려준다."""
    pairs, fail = [], []
    for d in docs:
        spans = by_text.get(d.get("text"))
        if spans is None:
            fail.append(d.get("id"))
            continue
        pairs.append((d, spans))
    print("%s : 성공 %s / 실패 %s" % (pad(what, 22), num(len(pairs)), num(len(fail))))
    if fail:
        print("  실패 문서 id (최대 20건): %s"
              % ", ".join(str(x) for x in fail[:20]))
    return pairs, fail


# ── 표 출력 ──────────────────────────────────────────────────
COLS = ("doc_id", "항목", "정답값", "OPF예측", "규칙예측", "판정")
WIDTHS = (18, 14, 22, 26, 26, 20)


def print_header():
    print(clip_line("  ".join(pad(h, w) for h, w in zip(COLS, WIDTHS))))
    print("-" * 78)


def print_case(row, text, s, e):
    """사례 1줄 + 문맥 1줄."""
    cells = [cut(v, w) for v, w in zip(row, WIDTHS)]
    print(clip_line("  ".join(pad(c, w) for c, w in zip(cells, WIDTHS)).rstrip()))
    print(clip_line("    " + context(text, s, e)))


def bucket_head(tag, title, total, shown, detail=""):
    print("")
    print("[%s] %s : %s건 중 %s건 표시%s"
          % (tag, pad(title, 28), num(total), num(shown), detail))


def rule_verdict(gs, ge, rspans):
    """규칙 레이어가 이 정답 스팬을 어떻게 다뤘는지."""
    if any(rs == gs and re_ == ge for rs, re_, _ in rspans):
        return "규칙정확"
    if hits(rspans, gs, ge):
        return "규칙부분"
    return "규칙미탐"


# ── 버킷 산출 ────────────────────────────────────────────────
def collect_gold_side(pairs, rule_by_text):
    """정답 스팬 축 버킷(A/B/F)의 모집단을 만든다.

    미탐 = 정답 스팬과 1자도 겹치는 OPF 예측이 없음.
    반환 원소: (doc_id, start, doc, gold, opf_spans, rule_spans)
    """
    miss = []          # OPF 완전 미탐 정답 스팬 전체
    for doc, opf in pairs:
        text = doc.get("text", "")
        rule = rule_by_text.get(text, [])
        did = doc.get("id")
        for g in doc.get("spans", []):
            gs, ge = int(g["start"]), int(g["end"])
            if hits(opf, gs, ge):
                continue
            miss.append((did, gs, doc, g, opf, rule))
    miss.sort(key=lambda x: (str(x[0]), x[1]))
    return miss


def collect_pred_side(pairs):
    """예측 스팬 축 버킷(C/D)의 모집단을 만든다.

    C 경계 과확장 : 어떤 정답을 통째로 덮되(ps<=gs, ge<=pe) 초과 문자가 있음
    D 순수 오탐   : 어떤 정답과도 1자도 겹치지 않음
    """
    over, pure = [], []
    for doc, opf in pairs:
        text = doc.get("text", "")
        did = doc.get("id")
        golds = doc.get("spans", [])
        for ps, pe, lb in opf:
            covered = [g for g in golds
                       if ps <= int(g["start"]) and int(g["end"]) <= pe]
            if covered:
                extra = (pe - ps) - max(int(g["end"]) - int(g["start"])
                                        for g in covered)
                if extra > 0:
                    widest = max(covered,
                                 key=lambda g: int(g["end"]) - int(g["start"]))
                    over.append((did, ps, doc, (ps, pe, lb), widest, extra))
                continue
            if not any(overlaps(ps, pe, int(g["start"]), int(g["end"]))
                       for g in golds):
                pure.append((did, ps, doc, (ps, pe, lb), None, 0))
    over.sort(key=lambda x: (str(x[0]), x[1]))
    pure.sort(key=lambda x: (str(x[0]), x[1]))
    return over, pure


def main():
    ap = argparse.ArgumentParser(
        prog="dump_examples.py",
        description="OPF 예측 · 규칙 예측 · 정본 대조 — 발표용 사례를 6버킷으로 선별해 "
                    "콘솔에 출력한다 (파일 출력 없음).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="선별에 난수를 쓰지 않는다. (doc_id, span.start) 정렬 후 앞에서부터 취하므로\n"
               "같은 입력이면 같은 사례가 같은 순서로 나온다.\n"
               "조인은 text 완전 일치로만 하며 실패가 1건이라도 있으면 exit 2 로 중단한다.")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--gold", default=os.path.join(root, "data",
                                                   "ss_pii_testset_ko_v1.json"),
                    help="정본 JSON (기본: eval/data/ss_pii_testset_ko_v1.json)")
    # required=True 로 두면 argparse 가 exit 2 로 끝나는데, exit 2 는 조인 실패
    # 전용 코드다. 인자 부재는 입력 오류이므로 직접 검사해 exit 1 로 맞춘다.
    ap.add_argument("--pred", default=None,
                    help="OPF 예측 JSONL (필수)")
    ap.add_argument("--rule", default=os.path.join(root, "results",
                                                   "rule_predictions.jsonl"),
                    help="규칙 예측 JSONL (기본: eval/results/rule_predictions.jsonl)")
    ap.add_argument("--probe-gold", default=None,
                    help="프로브 정본 JSONL (버킷 E, 선택)")
    ap.add_argument("--probe-pred", default=None,
                    help="프로브 OPF 예측 JSONL (버킷 E, 선택)")
    ap.add_argument("--rule-probe", default=None,
                    help="프로브 규칙 예측 JSONL (기본: --rule 파일명에 _probe 를 붙인 경로)")
    ap.add_argument("--n", type=int, default=5,
                    help="버킷당(A 는 항목당) 출력 건수 (기본 5). B/E 는 항목별 3건 고정")
    args = ap.parse_args()

    if not args.pred:
        die("--pred (OPF 예측 JSONL) 는 필수입니다. 예:\n"
            "  python3 %s --pred <OPF 예측>.jsonl" % os.path.basename(__file__), 1)
    if args.n < 1:
        die("--n 은 1 이상이어야 합니다 (받은 값: %d)" % args.n, 1)
    n = args.n

    docs = load_gold(args.gold)
    opf_by_text, n_opf = load_pred_by_text(args.pred, "OPF 예측 JSONL")
    rule_by_text, n_rule = load_pred_by_text(args.rule, "규칙 예측 JSONL")

    section("[0] 입력")
    print("정본 JSON      : %s  (문서 %s건)" % (args.gold, num(len(docs))))
    print("OPF 예측       : %s  (레코드 %s건)" % (args.pred, num(n_opf)))
    print("규칙 예측      : %s  (레코드 %s건)" % (args.rule, num(n_rule)))
    print("버킷당 건수    : --n %d  (B/E 는 항목별 %d건 고정)" % (n, FIXED_QUOTA))

    # ── [1] 조인 — text 완전 일치만 ──────────────────────────
    section("[1] 조인 (text 완전 일치)")
    print("하니스 example_id 는 sha256 자동 생성값이라 정본 id 와 다르므로 text 로만 조인한다.")
    pairs, fail_opf = join(docs, opf_by_text, "정본 - OPF 예측")
    _, fail_rule = join(docs, rule_by_text, "정본 - 규칙 예측")

    probe_docs = probe_pairs = None
    fail_probe_opf = fail_probe_rule = []
    if args.probe_gold or args.probe_pred:
        if not (args.probe_gold and args.probe_pred):
            die("--probe-gold 와 --probe-pred 는 함께 지정해야 합니다 (버킷 E).", 1)
        rule_probe = args.rule_probe
        if rule_probe is None:
            base, ext = os.path.splitext(args.rule)
            rule_probe = base + "_probe" + ext
        probe_docs = load_probe_gold(args.probe_gold)
        probe_opf, _ = load_pred_by_text(args.probe_pred, "프로브 OPF 예측 JSONL")
        probe_rule, _ = load_pred_by_text(rule_probe, "프로브 규칙 예측 JSONL")
        print("프로브 정본    : %s  (문서 %s건)" % (args.probe_gold, num(len(probe_docs))))
        print("프로브 규칙    : %s" % rule_probe)
        probe_pairs, fail_probe_opf = join(probe_docs, probe_opf, "프로브 - OPF 예측")
        _, fail_probe_rule = join(probe_docs, probe_rule, "프로브 - 규칙 예측")
        probe_rule_by_text = probe_rule

    total_fail = (len(fail_opf) + len(fail_rule)
                  + len(fail_probe_opf) + len(fail_probe_rule))
    print("")
    print("조인 실패 합계 : %s건" % num(total_fail))
    if total_fail:
        print("*** 실패: 조인 실패 건수가 0이 아닙니다. 사례 선별을 중단합니다. ***")
        raise SystemExit(2)
    print("조인 실패 0건 — 계속 진행")

    # ── 모집단 산출 ──────────────────────────────────────────
    miss = collect_gold_side(pairs, rule_by_text)
    over, pure = collect_pred_side(pairs)

    section("[2] 버킷별 모집단")
    print("미탐 판정(재현율 축) : 정답 스팬과 1자도 겹치는 OPF 예측이 없음")
    print("과탐 판정(정밀도 축) : 예측 ⊆ 정답 을 만족하지 못한 예측")
    print("두 축은 모집단이 다르므로 합산하지 않는다.")
    print("")
    print("OPF 완전 미탐 정답 스팬 : %s건 (정본 스팬 %s건 중)"
          % (num(len(miss)),
             num(sum(len(d.get("spans", [])) for d in docs))))
    print("OPF 예측 스팬 총계      : %s건"
          % num(sum(len(p) for _, p in pairs)))

    # ── 버킷 A ───────────────────────────────────────────────
    a_pool = {c: [m for m in miss if m[3].get("corp_category") == c]
              for c in CTX_A}
    a_total = sum(len(v) for v in a_pool.values())
    a_shown = sum(min(n, len(v)) for v in a_pool.values())
    section("[A] OPF 미탐 — 문맥 의존 3항목")
    bucket_head("A", "OPF 미탐 문맥의존", a_total, a_shown,
                "  (항목별 최대 %d건)" % n)
    print("  항목별 모집단: %s" % " / ".join("%s %s건" % (c, num(len(a_pool[c])))
                                             for c in CTX_A))
    for c in CTX_A:
        print("")
        print("  -- %s : %s건 중 %s건 --" % (c, num(len(a_pool[c])),
                                            num(min(n, len(a_pool[c])))))
        print_header()
        for did, gs, doc, g, opf, rule in a_pool[c][:n]:
            ge = int(g["end"])
            text = doc.get("text", "")
            print_case((did, c, g.get("value", text[gs:ge]), NONE_MARK,
                        render_spans(text, hits(rule, gs, ge)),
                        "완전미탐/" + rule_verdict(gs, ge, rule)),
                       text, gs, ge)

    # ── 버킷 B ───────────────────────────────────────────────
    b_all = [m for m in miss if m[3].get("corp_group") == UNIQUE_GROUP]
    b_pool = {c: [m for m in b_all if m[3].get("corp_category") == c]
              for c in UNIQUE_CATS}
    b_shown = sum(min(FIXED_QUOTA, len(v)) for v in b_pool.values())
    section("[B] OPF 미탐 — 고유식별정보 (corp_group)")
    bucket_head("B", "OPF 미탐 고유식별정보", len(b_all), b_shown,
                "  (4항목 각 %d건)" % FIXED_QUOTA)
    print("  항목별 모집단: %s" % " / ".join("%s %s건" % (c, num(len(b_pool[c])))
                                             for c in UNIQUE_CATS))
    for c in UNIQUE_CATS:
        print("")
        print("  -- %s : %s건 중 %s건 --" % (c, num(len(b_pool[c])),
                                            num(min(FIXED_QUOTA, len(b_pool[c])))))
        print_header()
        for did, gs, doc, g, opf, rule in b_pool[c][:FIXED_QUOTA]:
            ge = int(g["end"])
            text = doc.get("text", "")
            print_case((did, c, g.get("value", text[gs:ge]), NONE_MARK,
                        render_spans(text, hits(rule, gs, ge)),
                        "완전미탐/" + rule_verdict(gs, ge, rule)),
                       text, gs, ge)

    # ── 버킷 C ───────────────────────────────────────────────
    section("[C] 과탐 — 경계 과확장 (예측 ⊃ 정답, 초과 문자 있음)")
    bucket_head("C", "과탐 경계과확장", len(over), min(n, len(over)))
    print_header()
    for did, ps, doc, (ps_, pe, lb), g, extra in over[:n]:
        text = doc.get("text", "")
        gs, ge = int(g["start"]), int(g["end"])
        print_case((did, g.get("corp_category", "?"),
                    g.get("value", text[gs:ge]),
                    "%s[%s]" % (flat(text[ps_:pe]), lb or "?"),
                    render_spans(text, hits(rule_by_text.get(text, []), ps_, pe)),
                    "경계과확장(+%d자)" % extra),
                   text, ps_, pe)

    # ── 버킷 D ───────────────────────────────────────────────
    section("[D] 과탐 — 순수 오탐 (정답과 1자도 겹치지 않음)")
    bucket_head("D", "과탐 순수오탐", len(pure), min(n, len(pure)))
    print_header()
    for did, ps, doc, (ps_, pe, lb), _g, _x in pure[:n]:
        text = doc.get("text", "")
        print_case((did, lb or "?", NONE_MARK,
                    "%s[%s]" % (flat(text[ps_:pe]), lb or "?"),
                    render_spans(text, hits(rule_by_text.get(text, []), ps_, pe)),
                    "순수오탐"),
                   text, ps_, pe)

    # ── 버킷 E ───────────────────────────────────────────────
    section("[E] 프로브셋 — 규칙 0건 + OPF 미탐")
    if probe_pairs is None:
        print("--probe-gold / --probe-pred 미지정 — 버킷 E 를 건너뜁니다.")
    else:
        e_pool = {t: [] for t in PROBE_TYPES}
        e_other = []
        for doc, opf in probe_pairs:
            text = doc.get("text", "")
            rule = probe_rule_by_text.get(text, [])
            if rule:                      # 규칙 예측 0건 조건
                continue
            for g in doc.get("spans", []):
                gs, ge = int(g["start"]), int(g["end"])
                if hits(opf, gs, ge):     # OPF 도 미탐이어야 함
                    continue
                item = (doc.get("id"), gs, doc, g, opf, rule)
                (e_pool[doc["probe_type"]] if doc["probe_type"] in e_pool
                 else e_other).append(item)
        for t in PROBE_TYPES:
            e_pool[t].sort(key=lambda x: (str(x[0]), x[1]))
        e_total = sum(len(v) for v in e_pool.values()) + len(e_other)
        e_shown = sum(min(FIXED_QUOTA, len(e_pool[t])) for t in PROBE_TYPES)
        bucket_head("E", "프로브 양측 미탐", e_total, e_shown,
                    "  (P1/P2/P3 각 %d건)" % FIXED_QUOTA)
        print("  유형별 모집단: %s%s"
              % (" / ".join("%s %s건" % (t, num(len(e_pool[t]))) for t in PROBE_TYPES),
                 ("  (유형 미상 %s건)" % num(len(e_other))) if e_other else ""))
        for t in PROBE_TYPES:
            print("")
            print("  -- %s : %s건 중 %s건 --" % (t, num(len(e_pool[t])),
                                                num(min(FIXED_QUOTA, len(e_pool[t])))))
            print_header()
            for did, gs, doc, g, opf, rule in e_pool[t][:FIXED_QUOTA]:
                ge = int(g["end"])
                text = doc.get("text", "")
                print_case((did, g.get("corp_category", "?"),
                            g.get("value", text[gs:ge]),
                            NONE_MARK, NONE_MARK, "양측미탐(%s)" % t),
                           text, gs, ge)

    # ── 버킷 F ───────────────────────────────────────────────
    f_pool = [m for m in miss
              if any(rs == m[1] and re_ == int(m[3]["end"])
                     for rs, re_, _ in m[5])]
    section("[F] 규칙 정확 탐지 + OPF 미탐 (두 레이어 나란히)")
    bucket_head("F", "규칙 단독 구제", len(f_pool), min(n, len(f_pool)))
    print("  규칙 예측 경계가 정답과 정확히 일치하고(시작·끝 모두), OPF 는 1자도 겹치지 않은 정답.")
    print_header()
    for did, gs, doc, g, opf, rule in f_pool[:n]:
        ge = int(g["end"])
        text = doc.get("text", "")
        print_case((did, g.get("corp_category", "?"),
                    g.get("value", text[gs:ge]), NONE_MARK,
                    render_spans(text, hits(rule, gs, ge)),
                    "규칙구제(경계일치)"),
                   text, gs, ge)

    print("")
    print("=" * 78)
    print("끝. 파일은 쓰지 않았다 (콘솔 출력 전용).")
    print("=" * 78)


if __name__ == "__main__":
    main()
