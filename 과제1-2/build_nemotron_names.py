#!/usr/bin/env python3
"""Nemotron-Personas-Korea parquet → nemotron_names.json (1주차 추출 스크립트 복원).

nvidia/Nemotron-Personas-Korea 에는 성명 전용 컬럼이 없다(데이터셋 카드:
"이름·성 … 필드들은 포함하지 않습니다"). 대신 professional_persona 가
"김지영 씨는 …" 처럼 성명으로 시작하는 문장이라, 그 선두에서 성명을 뽑는다.

추출 규칙은 기존 nemotron_names.json 의 extraction 키에 적힌 그대로다.
  대상 컬럼 : professional_persona
  정규식    : ^([가-힣]{2,4})\\s*씨(?:는|가|의|와|께서|와도|를)
  매칭      : 문자열 선두에서만 (re.match, search 아님)
  범위      : 상위 50,000행을 파일에 실린 순서대로

── 정렬과 동점 처리 ────────────────────────────────────────
names 는 빈도 내림차순이다. 빈도가 같으면 **먼저 등장한 행이 앞** 이며,
collections.Counter.most_common 이 삽입 순서를 유지하는 성질을 그대로 쓴다.
기존 산출물에서 30,612종 중 동점쌍이 30,578개라 이 규칙이 순서를 사실상
결정한다. 가나다순 같은 다른 규칙을 쓰면 전혀 다른 순서가 나온다.
name_freq 도 같은 순서로 직렬화한다(기존 파일에서 names 와 name_freq 의
키 순서가 완전히 일치한다).

── 지역(regions) ───────────────────────────────────────────
데이터셋 컬럼에 시도는 province, 시군구는 district 로 따로 들어 있고,
기존 산출물의 값은 "경기-화성시" 처럼 둘을 '-' 로 이은 형태다. 그래서
province + "-" + district 로 합쳐 상위 20개를 센다.
※ 이 결합 방식은 기존 산출물의 값 형태에서 **추론한 것**이며, parquet 이
   없어 실측 대조를 하지 못했다. province 컬럼이 "경기" 가 아니라 "경기도"
   형태로 들어 있으면 결과가 달라진다. 실행 시 원시 province 값 표본을
   함께 출력하므로 첫 실행에서 눈으로 확인할 것.

parquet 읽기에만 pyarrow(없으면 pandas)를 쓰고 나머지는 표준 라이브러리다.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TEXT_COLUMN = "professional_persona"
PROVINCE_COLUMN = "province"
DISTRICT_COLUMN = "district"
NAME_RE = re.compile(r"^([가-힣]{2,4})\s*씨(?:는|가|의|와|께서|와도|를)")
SAMPLE_ROWS = 50000
TOP_REGIONS = 20

SOURCE = "nvidia/Nemotron-Personas-Korea (CC BY 4.0)"
METHOD = "professional_persona 선두 정규식"
REASON = "성명 전용 컬럼 부재 (README: '이름·성 … 필드들은 포함하지 않습니다')"
NOTE_NO_ROMAN = "로마자 표기 컬럼이 데이터셋에 존재하지 않아 names_roman 은 포함하지 않음"
NOTE_ROMAN = "로마자 표기 컬럼이 존재하므로 names_roman 을 별도로 검토할 것"


def die(msg, code=1):
    sys.stderr.write("[build_nemotron_names] %s\n" % msg)
    raise SystemExit(code)


def read_columns(path, columns):
    """parquet 에서 필요한 컬럼만 읽는다. (열 이름 목록, {열: 값 리스트})"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pass
    else:
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            die("parquet 을 열 수 없습니다: %s (%s)" % (path, e))
        names = list(pf.schema_arrow.names)
        want = [c for c in columns if c in names]
        table = pq.read_table(path, columns=want)
        return names, {c: table.column(c).to_pylist() for c in want}

    try:
        import pandas as pd
    except ImportError:
        die("parquet 을 읽으려면 pyarrow 또는 pandas 가 필요합니다.\n"
            "  설치: python3 -m pip install pyarrow\n"
            "  (공용 장비이므로 전역 설치 대신 가상환경 안에서 설치하십시오:\n"
            "   .venv/bin/pip install pyarrow)")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        die("parquet 을 읽을 수 없습니다: %s (%s)" % (path, e))
    names = list(df.columns)
    return names, {c: df[c].tolist() for c in columns if c in names}


def main():
    ap = argparse.ArgumentParser(
        prog="build_nemotron_names.py",
        description="Nemotron parquet 에서 nemotron_names.json 을 만든다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--write 없이 실행하면 원본을 건드리지 않고 .new 로만 쓴다.")
    ap.add_argument("--parquet",
                    default=os.path.join(ROOT, "archive",
                                         "train-00000-of-00009.parquet"),
                    help="입력 parquet (기본: archive/train-00000-of-00009.parquet)")
    ap.add_argument("--out", default=os.path.join(HERE, "nemotron_names.json.new"),
                    help="출력 경로 (기본: 과제1-2/nemotron_names.json.new)")
    ap.add_argument("--write", action="store_true",
                    help="이 플래그가 있을 때만 과제1-2/nemotron_names.json 을 덮어쓴다")
    args = ap.parse_args()

    if not os.path.isfile(args.parquet):
        die("parquet 파일을 찾을 수 없습니다: %s\n"
            "  Nemotron-Personas-Korea 의 data/train-00000-of-00009.parquet 이 필요합니다.\n"
            "  내려받기: huggingface-cli download nvidia/Nemotron-Personas-Korea "
            "--repo-type dataset --include 'data/train-00000-of-00009.parquet'"
            % args.parquet)

    columns, data = read_columns(
        args.parquet, [TEXT_COLUMN, PROVINCE_COLUMN, DISTRICT_COLUMN])
    if TEXT_COLUMN not in data:
        die("parquet 에 %s 컬럼이 없습니다.\n  실제 컬럼(%d개): %s"
            % (TEXT_COLUMN, len(columns), ", ".join(columns)))

    # ── 성명 추출 (상위 SAMPLE_ROWS 행, 선두 매칭) ──────────
    texts = data[TEXT_COLUMN][:SAMPLE_ROWS]
    counter = collections.Counter()      # 삽입 순서 = 첫 등장 행 순서
    extracted = missed = 0
    for t in texts:
        m = NAME_RE.match(t) if isinstance(t, str) else None
        if m is None:
            missed += 1
            continue
        counter[m.group(1)] += 1
        extracted += 1

    ordered = counter.most_common()      # 빈도 내림차순, 동점은 첫 등장 순
    names = [n for n, _ in ordered]
    name_freq = collections.OrderedDict(ordered)

    # ── 지역 상위 20 (province-district) ────────────────────
    prov = data.get(PROVINCE_COLUMN, [])[:SAMPLE_ROWS]
    dist = data.get(DISTRICT_COLUMN, [])[:SAMPLE_ROWS]
    region_count = collections.Counter()
    for p, d in zip(prov, dist):
        if not d:
            continue
        region_count["%s-%s" % (p, d) if p else str(d)] += 1
    regions = [{"district": k, "count": c}
               for k, c in region_count.most_common(TOP_REGIONS)]

    # ── 로마자 컬럼 존재 여부는 실제 스키마로 판정한다 ──────
    roman_cols = [c for c in columns
                  if "roman" in c.lower() or "latin" in c.lower()]
    romanization_column = bool(roman_cols)

    out = collections.OrderedDict()
    out["source"] = SOURCE
    out["extraction"] = collections.OrderedDict([
        ("method", METHOD),
        ("regex", NAME_RE.pattern),
        ("reason", REASON),
        # 기존 산출물의 표기 규칙을 따라 데이터셋 안의 경로로 적는다
        ("shard", "data/%s" % os.path.basename(args.parquet)),
        ("extracted", extracted),
        ("missed", missed),
    ])
    out["sampled"] = len(texts)
    out["unique"] = len(names)
    out["names"] = names
    out["name_freq"] = name_freq
    out["regions"] = regions
    out["romanization_column"] = romanization_column
    out["note"] = NOTE_ROMAN if romanization_column else NOTE_NO_ROMAN

    dst = os.path.join(HERE, "nemotron_names.json") if args.write else args.out
    try:
        with open(dst, "w", encoding="utf-8") as f:
            # 기존 nemotron_names.json 은 끝에 개행이 없다. 바이트 단위로
            # 같은 파일을 내려고 여기서도 개행을 붙이지 않는다.
            json.dump(out, f, ensure_ascii=False, indent=1)
    except OSError as e:
        die("출력 파일을 쓸 수 없습니다: %s (%s)" % (dst, e))

    print("입력 parquet   : %s" % args.parquet)
    print("컬럼 %d개      : %s" % (len(columns), ", ".join(columns)))
    print("출력           : %s%s" % (dst, "  (--write: 원본 덮어씀)" if args.write
                                     else "  (원본 미변경)"))
    print("sampled        : {:,}행".format(len(texts)))
    print("extracted      : {:,} / missed : {:,}".format(extracted, missed))
    print("unique         : {:,}종".format(len(names)))
    print("상위 5종       : %s" % ", ".join("%s(%d)" % (n, name_freq[n])
                                            for n in names[:5]))
    print("romanization_column : %s%s"
          % (romanization_column, (" — %s" % ", ".join(roman_cols))
             if roman_cols else " (로마자 컬럼 없음)"))
    print("regions 상위 3 : %s"
          % ", ".join("%s(%d)" % (r["district"], r["count"]) for r in regions[:3]))
    # 지역 결합 방식은 추론이라 원시값을 같이 보여준다 — 첫 실행에서 확인할 것
    print("province 원시값 표본 : %s"
          % ", ".join(sorted({str(p) for p in prov[:200] if p})[:8]))
    print("district 원시값 표본 : %s"
          % ", ".join(sorted({str(d) for d in dist[:200] if d})[:8]))


if __name__ == "__main__":
    main()
