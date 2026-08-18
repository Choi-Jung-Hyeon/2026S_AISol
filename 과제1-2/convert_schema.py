#!/usr/bin/env python3
"""정본 JSON -> opf eval 하니스 스키마 JSONL 변환.

spans 리스트를 "<라벨>: <값>" 키의 매핑으로 바꾼다.
같은 라벨·값이 한 문서에 여러 번 나오면 오프셋을 한 키에 모은다.
--label 로 OPF 5라벨판 / 사내 11항목판을 고른다.
"""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--label", choices=["opf", "corp"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = "opf_label" if a.label == "opf" else "corp_category"
    docs = json.load(open(a.gold, encoding="utf-8"))["documents"]
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            spans = {}
            for s in d["spans"]:
                k = "%s: %s" % (s[key], s["value"])
                spans.setdefault(k, []).append([s["start"], s["end"]])
            m = d.get("meta", {})
            rec = {
                "text": d["text"],
                "spans": spans,
                "info": {
                    "id": d["id"],
                    "n_spans": len(d["spans"]),
                    "finance_context": m.get("finance_context"),
                    "difficulty_cases": m.get("difficulty_cases"),
                    "char_len": m.get("char_len"),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %s (%d docs)" % (a.out, len(docs)))

if __name__ == "__main__":
    main()
