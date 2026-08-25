#!/usr/bin/env python3
"""고유식별정보 4종만 정규식으로 탐지하는 규칙 레이어.

대상: 주민등록번호 / 외국인등록번호 / 여권번호 / 운전면허번호
이 4종은 형식이 법적으로 고정되어 정규식으로 확정 탐지가 가능하다.
계좌번호처럼 금융기관마다 자릿수·구분자가 다른 항목은 대상이 아니다.

검증번호를 탐지 조건으로 쓰지 않는 이유
  우리 테스트셋의 값은 가명처리 지침에 따라 검증번호·Luhn 이 의도적으로
  불일치하도록 무효화되어 있다(정본 pseudonymization.principles 참조).
  따라서 검증번호 통과를 탐지 조건에 넣으면 전건 미탐이 난다.
  형식 일치만으로 탐지하고, 검증번호는 보지 않는다.

주민등록번호와 외국인등록번호는 형식이 6자리-7자리로 동일하다.
뒷자리 선두 숫자로 구분하되(1~4 내국인 / 5~8 외국인), 그 밖의 값은
"고유식별정보(구분불가)" 로 단일 처리하고 건수를 따로 보고한다.

표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import re
import sys

RRN_LIKE = "주민등록번호"
FRN_LIKE = "외국인등록번호"
AMBIG = "고유식별정보(구분불가)"
PASSPORT = "여권번호"
DRIVER = "운전면허번호"
TARGETS = [RRN_LIKE, FRN_LIKE, PASSPORT, DRIVER, AMBIG]

# 한국어 문서라 조사가 바로 뒤에 붙는다("주민번호가", "890101-1234567은").
# 경계는 숫자·영문에 대해서만 잡고, 한글 조사는 경계로 인정한다.
# (?<![0-9]) / (?![0-9]) 로 숫자 연속만 차단하면 조사 결합 케이스를 살릴 수 있다.
RE_ID13 = re.compile(r"(?<![0-9])(\d{6})-(\d{7})(?![0-9])")
RE_PASSPORT = re.compile(r"(?<![0-9A-Za-z])([A-Z]{1,2}\d{7,8})(?![0-9A-Za-z])")
RE_DRIVER = re.compile(r"(?<![0-9])(\d{2}-\d{2}-\d{6}-\d{2})(?![0-9])")

TEXT_KEYS = ("text", "input_text", "input", "content", "document")


def die(msg):
    sys.stderr.write("[rule_layer] %s\n" % msg)
    raise SystemExit(2)


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


def load_jsonl_docs(path):
    """하니스 스키마 JSONL(프로브셋 등)을 정본 문서 형태로 바꾼다."""
    if not os.path.isfile(path):
        die("입력 JSONL 파일을 찾을 수 없습니다: %s" % path)
    docs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = r.get("text")
            if not isinstance(text, str):
                continue
            info = r.get("info") or {}
            spans = []
            raw = r.get("spans")
            if isinstance(raw, dict):
                for key, offs in raw.items():
                    label = key.split(":", 1)[0].strip()
                    for off in offs or []:
                        if isinstance(off, (list, tuple)) and len(off) >= 2:
                            spans.append({
                                "start": int(off[0]), "end": int(off[1]),
                                "value": text[int(off[0]):int(off[1])],
                                "corp_category": info.get("corp_category", label),
                            })
            docs.append({"id": info.get("id", "L%05d" % ln), "text": text,
                         "spans": spans})
    if not docs:
        die("입력 JSONL 에 레코드가 없습니다: %s" % path)
    return docs


def classify_id13(back7):
    """뒷자리 선두 숫자로 내/외국인을 가른다. 그 밖은 구분불가."""
    lead = back7[0]
    if lead in "1234":
        return RRN_LIKE
    if lead in "5678":
        return FRN_LIKE
    return AMBIG


def detect(text):
    """규칙 탐지 결과를 (start, end, category) 리스트로 돌려준다."""
    found = []
    for m in RE_ID13.finditer(text):
        found.append((m.start(), m.end(), classify_id13(m.group(2))))
    for m in RE_DRIVER.finditer(text):
        found.append((m.start(), m.end(), DRIVER))
    for m in RE_PASSPORT.finditer(text):
        found.append((m.start(), m.end(), PASSPORT))
    # 운전면허(2-2-6-2)가 주민형(6-7)과 겹치는 구간이 나오면 넓은 쪽을 남긴다
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out = []
    for s, e, c in found:
        if out and s < out[-1][1] and e <= out[-1][1]:
            continue
        out.append((s, e, c))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description="고유식별정보 4종 규칙 기반 탐지 레이어",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=os.path.join(
        root, "data", "ss_pii_testset_ko_v1.json"),
        help="정본 JSON 경로")
    ap.add_argument("--jsonl", default=None,
                    help="정본 대신 하니스 스키마 JSONL 을 입력으로 쓴다(프로브셋 등)")
    ap.add_argument("--out", default=os.path.join(
        root, "results", "rule_predictions.jsonl"),
        help="예측 스팬 JSONL 출력 경로 (opf eval --predictions-out 과 동일 포맷)")
    args = ap.parse_args()

    docs = load_jsonl_docs(args.jsonl) if args.jsonl else load_gold(args.gold)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        die("출력 디렉토리가 없습니다: %s" % out_dir)

    gold_n = dict((c, 0) for c in (RRN_LIKE, FRN_LIKE, PASSPORT, DRIVER))
    hit_n = dict((c, 0) for c in (RRN_LIKE, FRN_LIKE, PASSPORT, DRIVER))
    ambig_n = 0
    fp_n = 0
    fp_docs = set()
    pred_total = 0

    with open(args.out, "w", encoding="utf-8") as fo:
        for doc in docs:
            text = doc.get("text", "")
            preds = detect(text)
            pred_total += len(preds)
            fo.write(json.dumps({
                "example_id": doc.get("id"),
                "text": text,
                "predictions": [
                    {"start": s, "end": e, "label": c} for s, e, c in preds],
            }, ensure_ascii=False) + "\n")

            golds = [g for g in doc.get("spans", [])
                     if g.get("corp_category") in gold_n]
            for g in golds:
                gold_n[g["corp_category"]] += 1
                # 재현율 축: 정답 스팬 ⊆ 예측 스팬
                if any(s <= g["start"] and g["end"] <= e for s, e, _ in preds):
                    hit_n[g["corp_category"]] += 1

            all_gold = doc.get("spans", [])
            for s, e, c in preds:
                if c == AMBIG:
                    ambig_n += 1
                # 오탐: 어떤 정답 스팬과도 겹치지 않는 탐지
                if not any(s < gg["end"] and gg["start"] < e for gg in all_gold):
                    fp_n += 1
                    fp_docs.add(doc.get("id"))

    src = args.jsonl if args.jsonl else args.gold
    print("")
    print("=" * 74)
    print("규칙 레이어 단독 탐지 결과")
    print("=" * 74)
    print("입력      : %s" % src)
    print("문서 수   : %d" % len(docs))
    print("예측 스팬 : %d" % pred_total)
    print("출력      : %s" % args.out)

    print("")
    print("=" * 74)
    print("[1] 고유식별정보 4종 재현율 (정답 ⊆ 예측)")
    print("=" * 74)
    hdr = "%-14s  %-14s  %-14s" % ("항목", "탐지/정답", "미탐/정답")
    print(hdr)
    print("-" * len(hdr))
    tg = th = 0
    for c in (RRN_LIKE, FRN_LIKE, PASSPORT, DRIVER):
        g, h = gold_n[c], hit_n[c]
        tg += g
        th += h
        print("%-14s  %-14s  %-14s" % (c, "%d/%d" % (h, g), "%d/%d" % (g - h, g)))
    print("-" * len(hdr))
    print("%-14s  %-14s  %-14s" % ("합계", "%d/%d" % (th, tg),
                                   "%d/%d" % (tg - th, tg)))

    print("")
    print("=" * 74)
    print("[2] 구분불가 및 오탐")
    print("=" * 74)
    print("고유식별정보(구분불가) 처리 : %d/%d" % (ambig_n, pred_total))
    print("  (6자리-7자리 형식이나 뒷자리 선두가 1~8 이 아닌 경우)")
    print("오탐 (정답 스팬과 무관)     : %d/%d" % (fp_n, pred_total))
    print("오탐 발생 문서              : %d/%d" % (len(fp_docs), len(docs)))
    print("")


if __name__ == "__main__":
    main()
