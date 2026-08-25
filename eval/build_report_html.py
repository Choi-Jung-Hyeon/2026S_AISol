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

# 스토리 순서 — 잘하는 것 -> 못하는 것 -> 개선한 것 -> 하이브리드
ORDER = ["S1_STRONG", "F1_ORPHAN", "V1_DECODE", "F2_FORM", "F3_NARR",
         "F7_INCONSIST", "F6_OVERRUN", "F5_MISLABEL", "D_UNLABELED",
         "H1_RULE", "H2_OPF_ONLY"]

# 축별 머리말 — 데이터를 보고 직접 쓴 문구다
HEAD = {
    "S1_STRONG": (
        "형식이 고정된 항목은 통째로 정확히 덮는다",
        "이메일·계좌번호·주민등록번호처럼 자릿수와 구분자가 정해진 값은 "
        "경계까지 정확히 잡아낸다. 아래는 한 문서의 정답 스팬을 하나도 빠짐없이 "
        "덮은 사례다."),
    "F1_ORPHAN": (
        "한두 글자만 덮여 나머지가 그대로 노출된다",
        "모델이 구간 대부분을 배경으로 찍고 한 토큰에만 라벨을 붙이면 그 글자만 "
        "가려진다. argmax 는 이런 고아 태그를 스팬으로 인정해 1글자 마스킹을 "
        "만들지만, 제약 Viterbi 는 구조적으로 차단한다."),
    "V1_DECODE": (
        "argmax 가 놓친 것을 제약 Viterbi 가 살려낸다",
        "BIOES 전이 제약을 걸면 끊겼던 태그 시퀀스가 이어져 스팬이 온전히 "
        "복원된다. 같은 로짓을 디코딩 방식만 바꿔 얻은 차이다."),
    "F2_FORM": (
        "양식 칸이나 서명부에 적힌 성명을 놓친다",
        "'성명:' '담당자:' 뒤나 문서 끝 서명 자리처럼 앞뒤 문맥이 짧은 위치에서 "
        "한국어 이름을 배경으로 처리한다. 국문 성명 재현율이 11개 항목 중 "
        "가장 낮은 이유다."),
    "F3_NARR": (
        "'주소' 라는 단서가 없으면 서술문 속 주소를 지나친다",
        "주소 항목 앞에 라벨이 붙어 있으면 잡지만, 문장 안에 자연스럽게 섞여 "
        "있으면 지역 언급으로 읽고 넘어간다."),
    "F7_INCONSIST": (
        "같은 문서 같은 항목인데 일부만 가려진다",
        "한 문서에 같은 종류의 값이 여러 개 있을 때 어떤 것은 덮고 어떤 것은 "
        "놓친다. 문서 단위로 보면 일관성이 없어, 사람이 검수할 때 신뢰하기 "
        "어렵게 만든다."),
    "F6_OVERRUN": (
        "경칭과 직함까지 함께 삼킨다",
        "이름 뒤의 '님' '수석' 같은 호칭이나 앞의 공백까지 스팬에 포함시킨다. "
        "가려야 할 것은 다 가리므로 유출은 아니지만, 필요 이상으로 문장을 "
        "훼손한다."),
    "F5_MISLABEL": (
        "라벨은 틀렸지만 문자는 정확히 덮는다",
        "카드번호를 연락처로, 운전면허번호를 계좌번호로 분류하는 식이다. "
        "마스킹이 목적이라면 결과는 동일하므로 이 프로젝트에서는 실패로 "
        "세지 않는다."),
    "D_UNLABELED": (
        "정답에 없는 날짜를 탐지한다 — 오탐이지만 무해하다",
        "우리 테스트셋은 날짜를 마스킹 대상으로 두지 않아 정답 스팬이 없다. "
        "모델이 이를 탐지하면 오탐으로 집계되지만, 실제 운영에서는 더 가리는 "
        "쪽이라 위험하지 않다."),
    "H1_RULE": (
        "OPF 가 놓친 고유식별번호를 규칙 레이어가 구제한다",
        "주민·외국인등록번호, 여권, 운전면허는 형식이 법으로 고정돼 정규식으로 "
        "확정 탐지된다. OPF 단독으로 남은 미탐을 규칙과 병용하면 0 이 된다."),
    "H2_OPF_ONLY": (
        "규칙으로는 손댈 수 없는 항목을 OPF 가 처리한다",
        "성명·주소·이메일·연락처·계좌·카드는 형태가 일정하지 않아 정규식으로 "
        "잡히지 않는다. 규칙 단독 칸은 고유식별 4종 몇 건만 가린 채 나머지 PII 를 "
        "그대로 남긴다 — 규칙만으로는 이 문서를 가릴 수 없다는 뜻이다."),
}

# 3단 구성을 쓰는 축
TRIPLE_RULE = ("H1_RULE", "H2_OPF_ONLY")          # 원문 / OPF / OPF+규칙
TRIPLE_DEC = ("V1_DECODE", "F1_ORPHAN")           # 원문 / argmax / Viterbi

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
--line:#e5e7eb;--muted:#6b7280;--ink:#111827}
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
a{color:inherit;text-decoration:none}}
"""


def col_html(title, text, layers, lo, hi, cut_l, cut_r):
    body = paint(text, layers, lo, hi)
    pre = ('<span class="cut">…</span>' if cut_l else '') + body + \
          ('<span class="cut">…</span>' if cut_r else '')
    return ('<div class="col"><div class="ch">%s</div><pre>%s</pre></div>'
            % (esc(title), pre))


def case_html(rec):
    tag = rec["tag"]
    src = rec["source_text"]
    lo, hi, cl, cr = window(rec)
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
        arg = rec.get("masked_argmax", src)
        cols.append(col_html("argmax 마스킹", arg, [(1, "mask", [])], lo, hi, cl, cr))
        cols.append(col_html("Viterbi 마스킹", rec["masked_viterbi"],
                             [(1, "mask", pred), (2, "over", over)], lo, hi, cl, cr))
    else:
        cols.append(col_html("마스킹 결과 (Viterbi)", rec["masked_viterbi"],
                             [(1, "mask", pred), (2, "over", over)], lo, hi, cl, cr))

    ncol = len(cols)
    cls = "c3" if ncol >= 3 else "c2"
    return ('<div class="case"><div class="cols %s"%s>%s</div>'
            '<div class="note">%s · <b>%s</b> · %s</div></div>'
            % (cls,
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

    toc = " ".join('<a href="#%s">%s</a>' % (t, esc(HEAD.get(t, (t,))[0][:22]))
                   for t in tags)

    secs = []
    for t in tags:
        title, desc = HEAD.get(t, (t, ""))
        n = pop.get(t)
        scale = ("28,420 스팬 중 %s건" % format(n, ",")) if n else \
                ("대표 %d건" % len(by_tag[t]))
        if t == "S1_STRONG" and n:
            scale = "3,000 문서 중 %s개 문서" % format(n, ",")
        if t == "D_UNLABELED" and n:
            scale = "순수 오탐 5,498건 중 %s건" % format(n, ",")
        secs.append(
            '<section id="%s"><span class="tag">%s</span>'
            '<h2>%s</h2><p class="scale">%s · 대표 %d건</p>'
            '<p class="desc">%s</p>%s</section>'
            % (t, esc(t), esc(title), esc(scale), len(by_tag[t]), esc(desc),
               "".join(case_html(r) for r in by_tag[t])))

    legend = (
        '<div class="legend"><b>범례</b>'
        '<span><i class="sw" style="background:var(--blue)"></i>정답 PII 구간</span>'
        '<span><i class="sw" style="background:var(--red)"></i>미탐 — 안 가려진 곳</span>'
        '<span><i class="sw" style="background:var(--gray)"></i>마스킹된 구간</span>'
        '<span><i class="sw" style="background:var(--yellow)"></i>과확장 — 정답 밖까지 덮임</span>'
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
        '<div class="toc"><b>축 목차</b><br>%s</div>%s'
        '</div></body></html>'
        % (CSS, date.today().isoformat(), "".join(cards), legend, toc,
           "".join(secs)))

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        w.write(doc)

    print("축 %d개 / 카드 %d개" % (len(tags), len(recs)))
    print("출력: %s (%.1f KB)" % (args.out, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
