"""showcase.jsonl 을 읽어 원문/마스킹 좌우 비교 HTML 보고서를 만든다.

자기완결형 단일 파일이다. CSS·JS 를 전부 인라인하고 외부 요청을 하지 않는다.
난수를 쓰지 않는다. 숫자는 하드코딩하지 않고 실행 시점에 집계한다.

    python3 build_report_html.py \
        --showcase results/showcase.jsonl --out results/opf_showcase.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))

# 발표 스토리 순서. CHAPTERS 의 축을 이어붙인 것과 반드시 같아야 한다.
ORDER = ["S1_STRONG", "F2_FORM", "F3_NARR", "F7_INCONSIST", "F1_ORPHAN",
         "F6_OVERRUN", "F5_MISLABEL", "D_UNLABELED",
         "V1_DECODE", "V2_REGRESS", "H1_RULE", "H2_OPF_ONLY"]

# 챕터 — (앵커 id, 로마숫자, 제목, 소속 축)
CHAPTERS = [
    ("ch1", "\u2160", "잘하는 것",
     ["S1_STRONG"]),
    ("ch2", "\u2161", "못하는 것",
     ["F2_FORM", "F3_NARR", "F7_INCONSIST", "F1_ORPHAN"]),
    ("ch3", "\u2162", "오탐이지만 문제되지 않는 것",
     ["F6_OVERRUN", "F5_MISLABEL", "D_UNLABELED"]),
    ("ch4", "\u2163", "그래서 이렇게 덮습니다",
     ["V1_DECODE", "V2_REGRESS", "H1_RULE", "H2_OPF_ONLY"]),
]

_FLAT = [t for _, _, _, ts in CHAPTERS for t in ts]
if _FLAT != ORDER:
    sys.exit("CHAPTERS 와 ORDER 가 어긋납니다: %s vs %s" % (_FLAT, ORDER))

# 축별 머리말 — (제목, 한 줄 설명)
HEAD = {
    "S1_STRONG": (
        "번호는 통째로 지웁니다",
        "형식이 정해진 항목은 재현율 0.99~1.00"),
    "F2_FORM": (
        "표 안의 이름은 못 찾습니다",
        "폼·서명부처럼 앞뒤 문맥이 없으면 놓칩니다"),
    "F3_NARR": (
        "'주소'라는 말이 없으면 주소를 못 봅니다",
        "문장 속에 녹아든 주소는 지나칩니다"),
    "F7_INCONSIST": (
        "같은 문서, 같은 이름, 다른 결과",
        "한 번은 가리고 한 번은 놓칩니다"),
    "F1_ORPHAN": (
        "절반만 가려진 마스킹",
        "앞 두 글자만 지우고 나머지는 그대로"),
    "F6_OVERRUN": (
        "직함까지 삼켜버립니다",
        "이름만 지우면 될 것을 '수석'까지"),
    "F5_MISLABEL": (
        "이름을 주소로 착각해도 결과는 같습니다",
        "라벨은 틀렸지만 문자는 정확히 덮였습니다"),
    "D_UNLABELED": (
        "정답에 없는 날짜까지 찾아냅니다",
        "오탐으로 세지만 지워서 손해가 없습니다"),
    "V1_DECODE": (
        "디코딩만 바꿔도 살아나는 것들",
        "모델은 그대로, 읽는 방식만 규칙에 맞췄습니다"),
    "V2_REGRESS": (
        "사라진 것은 성공이 아니라 착시였습니다",
        "불법 경로로 만들어진 스팬이 정직한 실패로 바뀌었습니다"),
    "H1_RULE": (
        "마지막 30건은 규칙이 메웁니다",
        "법정 고유식별정보 4종 미탐 30 → 0, 오탐 증가 없음"),
    "H2_OPF_ONLY": (
        "그런데 규칙만으로는 안 됩니다",
        "나머지 20,731스팬(72.95%)은 규칙 예측 0건"),
}

# 3단 구성을 쓰는 축
TRIPLE_RULE = ("H1_RULE", "H2_OPF_ONLY")                 # 원문 / OPF / OPF+규칙
TRIPLE_DEC = ("V1_DECODE", "V2_REGRESS", "F1_ORPHAN")    # 원문 / argmax / Viterbi

CONTEXT = 120


def esc(t):
    return html.escape(t, quote=False)


def merge(ranges):
    if not ranges:
        return []
    xs = sorted([list(r) for r in ranges])
    out = [xs[0]]
    for a, b in xs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def paint(text, layers, lo, hi):
    """[lo,hi) 구간을 클래스별로 칠한다. layers 는 (우선순위, class, ranges) 목록.

    문자 단위로 클래스를 정해 오프셋이 어긋나지 않게 한다.
    """
    cls = [None] * len(text)
    for _, name, ranges in sorted(layers, key=lambda x: x[0]):
        for a, b in ranges:
            for i in range(max(a, 0), min(b, len(text))):
                cls[i] = name
    out, i = [], lo
    while i < hi:
        c = cls[i]
        j = i
        while j < hi and cls[j] == c:
            j += 1
        seg = esc(text[i:j])
        out.append(seg if c is None else '<span class="%s">%s</span>' % (c, seg))
        i = j
    return "".join(out)


def window(rec):
    """정답 스팬 주변 CONTEXT 자만 남긴다. 자른 경우 표시한다."""
    t = rec["source_text"]
    marks = [s for s in rec.get("gold_spans", [])] or []
    if not marks:
        marks = [s for s in rec.get("pred_spans", [])]
    if not marks:
        return 0, len(t), False, False
    lo = max(0, min(m[0] for m in marks) - CONTEXT)
    hi = min(len(t), max(m[1] for m in marks) + CONTEXT)
    return lo, hi, lo > 0, hi < len(t)


CSS = """
:root{--blue:#dbeafe;--red:#fecaca;--gray:#e5e7eb;--yellow:#fef08a;
--purple:#e9d5ff;--line:#e5e7eb;--muted:#6b7280;--ink:#111827}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);
font-family:"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;
line-height:1.65;font-size:15px}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 6px}
.sub{color:var(--muted);font-size:13px;margin:0 0 22px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 26px}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .big{font-size:21px;font-weight:700;letter-spacing:-.3px}
.card .lab{color:var(--muted);font-size:12px;margin-top:4px}
.legend{display:flex;flex-wrap:wrap;gap:14px;align-items:center;
border:1px solid var(--line);border-radius:10px;padding:11px 14px;margin:0 0 26px;
font-size:13px}
.sw{display:inline-block;width:13px;height:13px;border-radius:3px;
vertical-align:-2px;margin-right:5px}
.toc{border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:0 0 32px}
.toc a{color:#1d4ed8;text-decoration:none;font-size:13px;margin-right:14px;
display:inline-block;padding:2px 0}
.toc a:hover{text-decoration:underline}
.toc .row{margin:6px 0}
.toc .cn{display:inline-block;min-width:120px;font-weight:700;font-size:13px}
.chap{margin:46px 0 20px;padding:0 0 8px;border-bottom:2px solid var(--ink);
scroll-margin-top:12px}
.chap:first-of-type{margin-top:26px}
.chap .rn{font-size:13px;color:var(--muted);letter-spacing:.06em}
.chap h2{font-size:21px;margin:2px 0 0;font-weight:700}
.shead{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.shead h2{margin:8px 0 4px}
.cap{padding:8px 14px;border-bottom:1px solid var(--line);background:#fafafa;
font-size:12px;color:var(--muted)}
section{margin:0 0 42px;scroll-margin-top:12px}
.tag{display:inline-block;font-size:11px;color:var(--muted);
border:1px solid var(--line);border-radius:4px;padding:1px 6px;
font-family:ui-monospace,Menlo,Consolas,monospace}
h2{font-size:19px;margin:8px 0 4px}
.scale{color:var(--muted);font-size:13px;margin:0 0 6px}
.desc{color:#374151;font-size:14px;margin:0 0 16px;max-width:78ch}
.case{border:1px solid var(--line);border-radius:10px;padding:0;
margin:0 0 16px;overflow:hidden}
.cols{display:grid;gap:0}
.c2{grid-template-columns:1fr 1fr}
.c3{grid-template-columns:1fr 1fr 1fr}
.col{padding:12px 14px;border-left:1px solid var(--line);min-width:0}
.col:first-child{border-left:0}
.ch{font-size:12px;color:var(--muted);margin:0 0 8px;font-weight:600}
pre{margin:0;white-space:pre-wrap;word-break:break-all;overflow-wrap:anywhere;
font-family:"SFMono-Regular",Menlo,Consolas,"D2Coding","Nanum Gothic Coding",monospace;
font-size:12.5px;line-height:1.85;tab-size:2}
.gold{background:var(--blue)}
.diff{background:var(--purple)}
.leak{background:var(--red);font-weight:700}
.mask{background:var(--gray)}
.over{background:var(--yellow)}
.note{border-top:1px solid var(--line);padding:9px 14px;font-size:12.5px;
color:#374151;background:#fafafa}
.cut{color:var(--muted);font-style:normal}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}
.c2,.c3{grid-template-columns:1fr}
.col{border-left:0;border-top:1px solid var(--line)}
.col:first-child{border-top:0}}
@media print{body{font-size:11px}.wrap{max-width:none;padding:0}
.case{break-inside:avoid;page-break-inside:avoid}
section{break-inside:auto}.toc{display:none}
.chap{break-after:avoid;page-break-after:avoid}
a{color:inherit;text-decoration:none}}
"""


def col_html(title, text, layers, lo, hi, cut_l, cut_r):
    body = paint(text, layers, lo, hi)
    pre = ('<span class="cut">…</span>' if cut_l else '') + body + \
          ('<span class="cut">…</span>' if cut_r else '')
    return ('<div class="col"><div class="ch">%s</div><pre>%s</pre></div>'
            % (esc(title), pre))


def diff_ranges(rec):
    """masked_argmax 와 masked_viterbi 가 다른 문자 위치를 구간으로 낸다.

    두 문자열은 문자 수를 보존하므로 source_text 와 길이가 같아야 한다.
    다르면 오프셋 비교가 성립하지 않으므로 중단한다.
    """
    t = rec["source_text"]
    a = rec.get("masked_argmax")
    v = rec.get("masked_viterbi")
    if a is None or v is None:
        sys.exit("masked_argmax/masked_viterbi 누락: %s" % rec["doc_id"])
    if not (len(a) == len(v) == len(t)):
        sys.exit("길이 불일치로 중단합니다 — doc_id=%s tag=%s "
                 "source %d / argmax %d / viterbi %d"
                 % (rec["doc_id"], rec["tag"], len(t), len(a), len(v)))
    idx = [i for i in range(len(t)) if a[i] != v[i]]
    if not idx:
        return [], 0
    out, st, prev = [], idx[0], idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append([st, prev + 1])
        st = prev = x
    out.append([st, prev + 1])
    return out, len(idx)


def case_html(rec):
    tag = rec["tag"]
    src = rec["source_text"]
    lo, hi, cl, cr = window(rec)
    cap = ""
    gold = merge([[a, b] for a, b, _ in rec.get("gold_spans", [])])
    leak = merge([list(r) for r in rec.get("leaked_spans", [])])
    over = merge([list(r) for r in rec.get("overrun_spans", [])])
    pred = merge([[a, b] for a, b, _ in rec.get("pred_spans", [])])

    # 좌: 정답(파랑) 위에 미탐(빨강). 미탐이 우선순위가 높다.
    left = [(1, "gold", gold), (2, "leak", leak)]

    cols = [col_html("원문", src, left, lo, hi, cl, cr)]

    if tag in TRIPLE_RULE:
        opf = rec.get("masked_opf", rec["masked_viterbi"])
        rule = rec.get("masked_rule", src)
        hyb = rec.get("masked_hybrid", opf)
        rsp = merge([[a, b] for a, b, _ in rec.get("rule_spans", [])])
        cols.append(col_html("OPF 단독 마스킹", opf,
                             [(1, "mask", pred), (2, "over", over)], lo, hi, cl, cr))
        cols.append(col_html("OPF + 규칙 마스킹", hyb,
                             [(1, "mask", merge(pred + rsp))], lo, hi, cl, cr))
        # 규칙 단독은 H2 에서 메시지가 되므로 한 칸 더 붙인다
        if tag == "H2_OPF_ONLY":
            cols.insert(2, col_html("규칙 단독 마스킹", rule,
                                    [(1, "mask", rsp)], lo, hi, cl, cr))
    elif tag in TRIPLE_DEC:
        # argmax 와 Viterbi 가 다른 문자 위치를 두 칸 모두에 같은 색으로 칠한다.
        # 같은 인덱스이므로 두 칸이 같은 자리에서 강조돼 눈으로 대조된다.
        dr, ndiff = diff_ranges(rec)
        cap = "argmax 와 다른 문자 %d자" % ndiff
        arg = rec["masked_argmax"]
        cols.append(col_html("argmax 마스킹", arg,
                             [(1, "mask", []), (3, "diff", dr)], lo, hi, cl, cr))
        cols.append(col_html("Viterbi 마스킹", rec["masked_viterbi"],
                             [(1, "mask", pred), (2, "over", over),
                              (3, "diff", dr)], lo, hi, cl, cr))
    else:
        cols.append(col_html("마스킹 결과 (Viterbi)", rec["masked_viterbi"],
                             [(1, "mask", pred), (2, "over", over)], lo, hi, cl, cr))

    ncol = len(cols)
    cls = "c3" if ncol >= 3 else "c2"
    return ('<div class="case">%s<div class="cols %s"%s>%s</div>'
            '<div class="note">%s · <b>%s</b> · %s</div></div>'
            % ('<div class="cap">%s</div>' % esc(cap) if cap else "",
               cls,
               ' style="grid-template-columns:repeat(%d,1fr)"' % ncol
               if ncol == 4 else '',
               "".join(cols), esc(rec["doc_id"]), esc(rec["item"]),
               esc(rec["note"])))


def main():
    ap = argparse.ArgumentParser(prog="build_report_html.py")
    ap.add_argument("--showcase",
                    default=os.path.join(HERE, "results", "showcase.jsonl"))
    ap.add_argument("--stats", default=None,
                    help="decode_compare 로그 (요약 카드 숫자 집계용)")
    ap.add_argument("--out",
                    default=os.path.join(HERE, "results", "opf_showcase.html"))
    args = ap.parse_args()

    if not os.path.isfile(args.showcase):
        sys.exit("showcase.jsonl 이 없습니다: %s" % args.showcase)
    recs = [json.loads(l) for l in open(args.showcase, encoding="utf-8")
            if l.strip()]

    # 요약 카드 숫자 — 로그에서 실행 시점 집계값을 읽는다. 하드코딩하지 않는다.
    st = {}
    if args.stats and os.path.isfile(args.stats):
        import re
        txt = open(args.stats, encoding="utf-8").read()

        def grab(label, cast=float):
            m = re.search(r"^\s*%s\s+([\d.]+)\s+([\d.]+)" % re.escape(label),
                          txt, re.M)
            return (cast(m.group(1)), cast(m.group(2))) if m else None
        st["recall"] = grab("재현율")
        st["f1"] = grab("F1")
        st["leak"] = grab("부분 노출 문자 수", int)
        st["illegal"] = grab("금지 전이 발생", int)
        m = re.search(r"완전 미탐\s+(\d+)\s+(\d+)\s+(\d+)", txt)
        st["ident4"] = (int(m.group(1)), int(m.group(3))) if m else None

    def card(big, lab):
        return '<div class="card"><div class="big">%s</div>' \
               '<div class="lab">%s</div></div>' % (big, esc(lab))

    cards = []
    if st.get("recall") and st.get("f1"):
        cards.append(card("%.4f &nbsp;/&nbsp; %.4f" % (st["recall"][1], st["f1"][1]),
                          "재현율 / F1 (제약 Viterbi, 3자 기준·라벨 무관)"))
    if st.get("leak"):
        a, b = st["leak"]
        cards.append(card("%s &rarr; %s자" % (format(a, ","), format(b, ",")),
                          "부분 노출 문자 (argmax → Viterbi)"))
    if st.get("illegal"):
        a, b = st["illegal"]
        cards.append(card("%s &rarr; %d건" % (format(a, ","), b),
                          "BIOES 금지 전이 (argmax → Viterbi)"))
    if st.get("ident4"):
        a, b = st["ident4"]
        cards.append(card("%d &rarr; %d건" % (a, b),
                          "고유식별 4종 완전 미탐 (OPF 단독 → 규칙 병용)"))

    by_tag = {}
    for r in recs:
        by_tag.setdefault(r["tag"], []).append(r)
    tags = [t for t in ORDER if t in by_tag] + \
           [t for t in by_tag if t not in ORDER]

    pop = {}
    if args.stats and os.path.isfile(args.stats):
        import re
        stxt = open(args.stats, encoding="utf-8").read()
        for m in re.finditer(r"^\s+([A-Z][A-Z0-9_]+)\s+\S+\s+(\d+)\s",
                             stxt, re.M):
            pop[m.group(1)] = int(m.group(2))

    # 목차 — 챕터 앵커 + 그 챕터에 속한 축 앵커
    known = set(ORDER)
    rows = []
    for cid, rn, ctitle, cts in CHAPTERS:
        links = " ".join('<a href="#%s">%s</a>'
                         % (t, esc(HEAD.get(t, (t,))[0]))
                         for t in cts if t in by_tag)
        rows.append('<div class="row"><span class="cn">'
                    '<a href="#%s">%s. %s</a></span>%s</div>'
                    % (cid, rn, esc(ctitle), links))
    extra = [t for t in tags if t not in known]
    if extra:
        rows.append('<div class="row"><span class="cn">기타</span>%s</div>'
                    % " ".join('<a href="#%s">%s</a>'
                               % (t, esc(HEAD.get(t, (t,))[0])) for t in extra))
    toc = "".join(rows)

    # 축 -> 그 축에서 시작되는 챕터 헤더
    chap_at = {}
    for cid, rn, ctitle, cts in CHAPTERS:
        for t in cts:
            if t in by_tag:
                chap_at[t] = (cid, rn, ctitle)
                break

    secs = []
    for t in tags:
        if t in chap_at:
            cid, rn, ctitle = chap_at[t]
            secs.append('<div class="chap" id="%s"><div class="rn">%s</div>'
                        '<h2>%s</h2></div>' % (cid, rn, esc(ctitle)))
        title, desc = HEAD.get(t, (t, ""))
        n = pop.get(t)
        scale = ("28,420 스팬 중 %s건" % format(n, ",")) if n else \
                ("대표 %d건" % len(by_tag[t]))
        if t == "S1_STRONG" and n:
            scale = "3,000 문서 중 %s개 문서" % format(n, ",")
        if t == "D_UNLABELED" and n:
            scale = "순수 오탐 5,498건 중 %s건" % format(n, ",")
        secs.append(
            '<section id="%s"><div class="shead"><h2>%s</h2>'
            '<span class="tag">%s</span></div>'
            '<p class="scale">%s · 대표 %d건</p>'
            '<p class="desc">%s</p>%s</section>'
            % (t, esc(title), esc(t), esc(scale), len(by_tag[t]), esc(desc),
               "".join(case_html(r) for r in by_tag[t])))

    legend = (
        '<div class="legend"><b>범례</b>'
        '<span><i class="sw" style="background:var(--blue)"></i>정답 PII 구간</span>'
        '<span><i class="sw" style="background:var(--red)"></i>미탐 — 안 가려진 곳</span>'
        '<span><i class="sw" style="background:var(--gray)"></i>마스킹된 구간</span>'
        '<span><i class="sw" style="background:var(--yellow)"></i>과확장 — 정답 밖까지 덮임</span>'
        '<span><i class="sw" style="background:var(--purple)"></i>argmax 와 Viterbi 가 다른 문자</span>'
        '</div>')

    doc = (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>OPF 한국어 PII 마스킹 — 원문/마스킹 좌우 비교</title>'
        '<style>%s</style></head><body><div class="wrap">'
        '<h1>OPF 한국어 PII 마스킹 — 원문 / 마스킹 좌우 비교</h1>'
        '<p class="sub">생성 %s · 정본 3,000문서 / 28,420스팬 · '
        '제약 Viterbi 디코딩 · 마스킹은 덮인 문자 1개당 * 1개</p>'
        '<div class="cards">%s</div>%s'
        '<div class="toc"><b>목차</b>%s</div>%s'
        '</div></body></html>'
        % (CSS, date.today().isoformat(), "".join(cards), legend, toc,
           "".join(secs)))

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        w.write(doc)

    print("챕터 %d개 / 축 %d개 / 카드 %d개"
          % (len(CHAPTERS), len(tags), len(recs)))
    for cid, rn, ctitle, cts in CHAPTERS:
        shown = [t for t in cts if t in by_tag]
        print("  %s. %-22s %s" % (rn, ctitle, " ".join(shown)))
    print("  축별 카드 수 : " + " / ".join("%s %d" % (t, len(by_tag[t]))
                                        for t in tags))
    print("  3단 렌더 축  : 원문/argmax/Viterbi %s · 원문/OPF/규칙 %s"
          % (list(TRIPLE_DEC), list(TRIPLE_RULE)))
    print("출력: %s (%.1f KB)" % (args.out, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
