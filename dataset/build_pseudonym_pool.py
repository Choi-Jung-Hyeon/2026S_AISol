#!/usr/bin/env python3
"""nemotron_names.json → pseudonym_pool.json 재계산 (1주차 추출 스크립트 복원).

pseudonym_pool.json 9개 키 중 Nemotron 에서 다시 계산하는 것은 4개뿐이다.

  surnames         성씨를 출현 빈도로 집계 → 상위 40종
  surname_weights  각 성씨의 출현 비율(%). 소수 둘째 자리 반올림
  given            이름(성 제외 부분)을 출현 빈도로 집계 → 상위 300종
  source_stats     sampled / unique_names / unique_surnames / unique_given

나머지 5개 키는 Nemotron 에서 나오지 않으므로 --base 에서 그대로 승계한다.

  romanization / romanization_provenance / address / email_domains / note

Nemotron-Personas-Korea 에는 로마자 표기 컬럼이 없어 실측 교체가 불가능했고,
기존 관용 표기와 로마자 표기법 보충분이 romanization_provenance 에 항목별로
기록되어 있다. 여기서 새로 만들면 그 이력이 사라진다.

── 빈도 집계의 기준 두 가지 (기존 산출물로 확정한 것) ────────────
1. "출현 빈도" 는 고유 성명 수가 아니라 **등장 횟수**(name_freq 값)다.
   등장 횟수 50,000 을 분모로 써야 김 21.32 / 이 14.81 / 박 8.44 /
   최 4.90 / 정 4.87 과 합계 94.45 가 재현된다. 고유 성명 수를 쓰면
   김 11.91 이 되고 합계도 94.50 으로 어긋난다.
2. 성/이름 분리는 **복성 6종을 두 글자로 끊는다**. 첫 글자만 성으로 쓰면
   surname_weights 합계가 94.50, given 순서, unique_surnames(110/116),
   unique_given(7401/7366) 이 모두 어긋난다.

── 동점 처리 ────────────────────────────────────────────────
빈도가 같을 때는 **name_freq 의 키 순서(먼저 나온 것이 앞)** 를 따른다.
collections.Counter.most_common 이 삽입 순서를 유지하는 성질을 그대로 쓴다.
given 상위 300 안에는 동점쌍이 210개 있고 300위 경계 자체가 동점(33회)이라,
가나다순 같은 다른 규칙을 쓰면 어떤 항목이 300 안에 드는지가 달라진다.
따라서 이 스크립트의 출력은 입력 JSON 의 키 순서에 의존한다.

표준 라이브러리만 사용한다.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_SURNAMES = 40
N_GIVEN = 300

# 복성(두 글자 성씨). 기존 산출물의 unique_surnames 116 / unique_given 7366 을
# 재현하는 목록이다. names 안에는 '어금' 도 1건 있으나 이를 복성으로 끊으면
# unique_surnames 가 117 이 되어 기록값과 어긋나므로 넣지 않는다.
COMPOUND_SURNAMES = ("남궁", "황보", "제갈", "선우", "사공", "서문")

# --base 에서 그대로 승계하는 키 (Nemotron 에서 재계산하지 않는다)
INHERITED_KEYS = ("romanization", "romanization_provenance", "address",
                  "email_domains", "note")


def die(msg, code=1):
    sys.stderr.write("[build_pseudonym_pool] %s\n" % msg)
    raise SystemExit(code)


def load_json(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        die("%s JSON 파싱 실패: %s (%s)" % (what, path, e))
    except OSError as e:
        die("%s 을 읽을 수 없습니다: %s (%s)" % (what, path, e))


def split_name(name):
    """성명을 (성, 이름) 으로 끊는다. 복성은 두 글자, 나머지는 한 글자."""
    if len(name) >= 3 and name[:2] in COMPOUND_SURNAMES:
        return name[:2], name[2:]
    return name[:1], name[1:]


def main():
    ap = argparse.ArgumentParser(
        prog="build_pseudonym_pool.py",
        description="nemotron_names.json 에서 pseudonym_pool.json 을 재계산한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--write 없이 실행하면 원본을 건드리지 않고 .new 로만 쓴다.")
    ap.add_argument("--names", default=os.path.join(HERE, "nemotron_names.json"),
                    help="입력 nemotron_names.json (기본: dataset/nemotron_names.json)")
    ap.add_argument("--base", default=os.path.join(HERE, "pseudonym_pool.json"),
                    help="비Nemotron 키 5종을 승계할 기존 pseudonym_pool.json")
    ap.add_argument("--out", default=os.path.join(HERE, "pseudonym_pool.json.new"),
                    help="출력 경로 (기본: dataset/pseudonym_pool.json.new)")
    ap.add_argument("--write", action="store_true",
                    help="이 플래그가 있을 때만 --base 원본을 덮어쓴다")
    args = ap.parse_args()

    N = load_json(args.names, "nemotron_names.json")
    base = load_json(args.base, "기존 pseudonym_pool.json")

    name_freq = N.get("name_freq")
    names = N.get("names")
    if not isinstance(name_freq, dict) or not name_freq:
        die("nemotron_names.json 에 name_freq 가 없습니다: %s" % args.names)
    if not isinstance(names, list) or not names:
        die("nemotron_names.json 에 names 가 없습니다: %s" % args.names)

    # ── 빈도 집계 (등장 횟수 기준) ──────────────────────────
    # name_freq 를 파일에 적힌 순서대로 넣어 동점 순서를 보존한다.
    sur_count = collections.Counter()
    giv_count = collections.Counter()
    for name, cnt in name_freq.items():
        surname, given = split_name(name)
        if surname:
            sur_count[surname] += cnt
        if given:
            giv_count[given] += cnt
    total = sum(name_freq.values())

    surnames = [s for s, _ in sur_count.most_common(N_SURNAMES)]
    surname_weights = [round(100.0 * sur_count[s] / total, 2) for s in surnames]
    given = [g for g, _ in giv_count.most_common(N_GIVEN)]

    # ── source_stats (고유 성명 기준) ───────────────────────
    uniq_sur, uniq_giv = set(), set()
    for name in names:
        s, g = split_name(name)
        if s:
            uniq_sur.add(s)
        if g:
            uniq_giv.add(g)
    source_stats = {
        "sampled": N.get("sampled", total),
        "unique_names": N.get("unique", len(names)),
        "unique_surnames": len(uniq_sur),
        "unique_given": len(uniq_giv),
    }

    # ── 승계 키 확인 ────────────────────────────────────────
    missing_keys = [k for k in INHERITED_KEYS if k not in base]
    if missing_keys:
        die("--base 에 승계할 키가 없습니다: %s\n  기존 파일: %s"
            % (", ".join(missing_keys), args.base))

    # ── 로마자 표기 커버리지 검사 (build_dataset.py 가 기동 시 하는 assert) ──
    roman = base["romanization"]
    miss_sur = [s for s in surnames if s not in (roman.get("surname") or {})]
    miss_giv = [g for g in given if g not in (roman.get("given") or {})]
    if miss_sur or miss_giv:
        sys.stderr.write("[build_pseudonym_pool] 승계한 romanization 에 표기가 없는 항목이 "
                         "있습니다. build_dataset.py 가 기동 시 실패합니다.\n")
        if miss_sur:
            sys.stderr.write("  성씨 %d종: %s\n" % (len(miss_sur), ", ".join(miss_sur)))
        if miss_giv:
            sys.stderr.write("  이름 %d종: %s\n" % (len(miss_giv), ", ".join(miss_giv)))
        raise SystemExit(1)

    # ── 조립 (기존 파일의 키 순서를 그대로 따른다) ──────────
    computed = {
        "surnames": surnames,
        "surname_weights": surname_weights,
        "given": given,
        "source_stats": source_stats,
    }
    out = {}
    for k in base:
        out[k] = computed[k] if k in computed else base[k]
    for k in computed:            # 기존 파일에 없던 키가 있으면 뒤에 붙인다
        if k not in out:
            out[k] = computed[k]

    dst = args.base if args.write else args.out
    try:
        with open(dst, "w", encoding="utf-8") as f:
            # 기존 pseudonym_pool.json 은 끝에 개행이 없다. 바이트 단위로
            # 같은 파일을 내려고 여기서도 개행을 붙이지 않는다.
            json.dump(out, f, ensure_ascii=False, indent=1)
    except OSError as e:
        die("출력 파일을 쓸 수 없습니다: %s (%s)" % (dst, e))

    print("입력 names        : %s" % args.names)
    print("승계 base         : %s" % args.base)
    print("출력              : %s%s" % (dst, "  (--write: 원본 덮어씀)" if args.write
                                        else "  (원본 미변경)"))
    print("재계산 키 4종     : surnames %d / surname_weights %d / given %d / source_stats"
          % (len(surnames), len(surname_weights), len(given)))
    print("승계 키 5종       : %s" % ", ".join(INHERITED_KEYS))
    print("등장 횟수 합계    : {:,}".format(total))
    print("surname_weights 합: %.2f" % round(sum(surname_weights), 2))
    print("source_stats      : %s" % json.dumps(source_stats, ensure_ascii=False))
    print("로마자 커버리지   : 성씨 %d/%d, 이름 %d/%d — 누락 0"
          % (len(surnames), len(surnames), len(given), len(given)))


if __name__ == "__main__":
    main()
