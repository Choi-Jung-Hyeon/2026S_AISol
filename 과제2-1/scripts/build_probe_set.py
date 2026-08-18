#!/usr/bin/env python3
"""부분 발화 프로브셋 300건을 생성한다 (P1 절단 / P2 구어체 / P3 띄어쓰기).

실제 상담에서는 PII 가 끝까지 발화되지 않거나, 구어체 군더더기가 섞이거나,
구분자가 무너진 채로 나타난다. 정본 테스트셋에는 이 케이스가 없어 측정이
불가능하므로 별도 프로브셋으로 만든다. 이 케이스도 탐지되어야 한다는 것이
사내 요구사항이다.

값의 출처
  - 국문/영문 성명, 주소 : 과제1-2/pseudonym_pool.json 의 가명값 풀 조합
  - 이메일, 숫자형 7종   : 정본 테스트셋의 기존 가명값을 재사용(읽기 전용)
    풀에 숫자형 PII 값이 없기 때문이며, 실존 가능한 값을 새로 만들지 않기
    위해 이미 무효화된 가명값을 그대로 가져온다.

값 무효화
  주민/외국인등록번호는 검증번호가, 카드번호는 Luhn 이 통과하지 않아야 한다.
  정본의 외국인등록번호 중 일부는 구(舊) 외국인번호 검증식(+2 보정)을
  우연히 통과하므로, 표본 추출 단계에서 두 검증식을 모두 실패하는 값만 쓴다.

표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import random
import sys

SEED = 20260818

POOL_CATS = {"국문 성명", "영문 성명", "주소"}
NUMERIC_CATS = {"주민등록번호", "외국인등록번호", "여권번호", "운전면허번호",
                "계좌번호", "카드번호", "연락처"}

# 유형별 항목 배분 (합계 각 100)
ALLOC = {
    "P1": [("계좌번호", 20), ("주민등록번호", 20), ("카드번호", 15),
           ("연락처", 15), ("외국인등록번호", 10), ("운전면허번호", 10),
           ("주소", 10)],
    "P2": [("연락처", 25), ("카드번호", 20), ("계좌번호", 15),
           ("주민등록번호", 15), ("국문 성명", 15), ("이메일 주소", 10)],
    "P3": [("연락처", 25), ("카드번호", 20), ("계좌번호", 15),
           ("운전면허번호", 10), ("주민등록번호", 10), ("영문 성명", 10),
           ("이메일 주소", 10)],
}


def die(msg):
    sys.stderr.write("[build_probe_set] %s\n" % msg)
    raise SystemExit(2)


def load_json(path, what):
    if not os.path.isfile(path):
        die("%s 파일을 찾을 수 없습니다: %s" % (what, path))
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        die("%s 파일이 올바른 JSON 이 아닙니다: %s (%s)" % (what, path, e))


# ---------- 검증번호 ----------
def rrn_check_ok(v):
    """주민등록번호 검증식. 통과하면 True (= 프로브셋에 쓰면 안 되는 값)."""
    s = "".join(c for c in v if c.isdigit())
    if len(s) != 13:
        return False
    w = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    tot = sum(int(s[i]) * w[i] for i in range(12))
    return ((11 - tot % 11) % 10) == int(s[12])


def frn_check_ok(v):
    """구 외국인등록번호 검증식(+2 보정). 통과하면 True."""
    s = "".join(c for c in v if c.isdigit())
    if len(s) != 13:
        return False
    w = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    tot = sum(int(s[i]) * w[i] for i in range(12))
    return (((11 - tot % 11) % 10 + 2) % 10) == int(s[12])


def luhn_ok(v):
    """카드번호 Luhn. 통과하면 True."""
    s = [c for c in v if c.isdigit()]
    if not s:
        return False
    tot = 0
    for i, c in enumerate(reversed(s)):
        n = int(c)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        tot += n
    return tot % 10 == 0


def surface_ok(cat, surface):
    """발화 표면값이 프로브로 쓸 만한지 본다.

    절단 지점에 따라 '010', '333-' 처럼 숫자가 서너 개만 남는 경우가 생기는데,
    이는 PII 라기보다 잡음이라 미탐으로 세면 지표만 왜곡된다. 숫자형 항목은
    최소 4자리를 요구한다. 성명·주소는 3글자 값이 정상이므로 예외로 둔다.
    """
    s = surface.strip()
    if not s:
        return False
    if cat in NUMERIC_CATS:
        return sum(c.isdigit() for c in s) >= 4
    return len(s) >= 2


def value_is_invalidated(cat, v):
    if cat == "주민등록번호":
        return not rrn_check_ok(v)
    if cat == "외국인등록번호":
        return not rrn_check_ok(v) and not frn_check_ok(v)
    if cat == "카드번호":
        return not luhn_ok(v)
    return True


# ---------- 값 공급 ----------
def build_value_banks(gold_docs, pool, rng):
    """항목별 사용 가능한 가명값 목록을 만든다."""
    banks = {}
    seen = {}
    for doc in gold_docs:
        for sp in doc.get("spans", []):
            cat = sp.get("corp_category")
            if cat in NUMERIC_CATS or cat == "이메일 주소":
                seen.setdefault(cat, set()).add(sp.get("value", ""))
    # 표본 단계 제외는 더 이상 하지 않는다.
    # 정본이 주민식·구 외국인식 양쪽에서 무효화되도록 수정되면서
    # (구 외국인등록번호 검증식 통과 166/1580 건 해소) 걸러낼 값이 없어졌다.
    # 다만 회귀를 놓치지 않도록 "검증식을 통과하는 정본 값" 건수는 계속 센다.
    residual = {}
    for cat, vals in seen.items():
        keep = sorted(v for v in vals if v)
        residual[cat] = sum(1 for v in keep if not value_is_invalidated(cat, v))
        rng.shuffle(keep)
        banks[cat] = keep

    # 풀 기반 항목
    sur, giv = pool["surnames"], pool["given"]
    rs, rg = pool["romanization"]["surname"], pool["romanization"]["given"]
    ko = [s + g for s in sur for g in giv[:60]]
    rng.shuffle(ko)
    banks["국문 성명"] = ko
    en = ["%s %s" % (rg[g], rs[s]) for s in sur if s in rs
          for g in giv[:60] if g in rg]
    rng.shuffle(en)
    banks["영문 성명"] = en
    ad = pool["address"]
    addr = []
    for sido in ad["sido"]:
        for gugun in ad["gugun"].get(sido, []):
            for road in ad["road"]:
                addr.append("%s %s %s" % (sido, gugun, road))
    rng.shuffle(addr)
    banks["주소"] = addr
    return banks, residual


# ---------- 변형 ----------
SEPS = "- ."


def sep_positions(v):
    return [i for i, c in enumerate(v) if c in SEPS]


def truncate_value(v, cat, rng):
    """값을 중간에서 자른다. 구분자 유지 여부를 케이스별로 섞는다."""
    if cat == "주소":
        parts = v.split(" ")
        if len(parts) > 1:
            k = rng.randint(1, len(parts) - 1)
            return " ".join(parts[:k])
        return v
    seps = sep_positions(v)
    cands = []
    for i in seps:
        if i >= 4:
            cands.append(i)          # 구분자 버림  예) 977-18
        if i + 1 >= 4 and i + 1 < len(v):
            cands.append(i + 1)      # 구분자 유지  예) 977-18-
    for i in range(4, len(v)):       # 그룹 중간 절단
        if v[i - 1] not in SEPS:
            cands.append(i)
    cands = sorted(set(c for c in cands if 4 <= c < len(v)))
    if not cands:
        return v
    return v[:rng.choice(cands)]


COLLOQUIAL_JOINS = ["에 ", " 에 ", "이랑 ", " 다시 ", " "]


def colloquialize(v, rng):
    """구분자를 구어 연결어로 바꾼다. 예) 010에 1234에 5678"""
    seps = sep_positions(v)
    if not seps:
        return v
    join = rng.choice(COLLOQUIAL_JOINS)
    out, prev = [], 0
    for i in seps:
        out.append(v[prev:i])
        prev = i + 1
    out.append(v[prev:])
    return join.join(p for p in out if p)


def respace(v, rng):
    """구분자를 공백/없음/혼합으로 변형한다."""
    seps = sep_positions(v)
    if not seps:
        # 공백이 없는 값(여권 등)은 중간에 공백을 넣는다
        if len(v) > 4:
            i = rng.randint(2, len(v) - 2)
            return v[:i] + " " + v[i:]
        return v
    mode = rng.choice(["space", "none", "mixed"])
    out = []
    for i, c in enumerate(v):
        if i in seps:
            if mode == "space":
                out.append(" ")
            elif mode == "none":
                pass
            else:
                out.append(rng.choice([" ", "", c]))
        else:
            out.append(c)
    return "".join(out)


# ---------- 문맥 템플릿 ----------
P1_TPL = [
    ("제 %s가 ", " 이렇게 시작하는데 맞나요?"),
    ("%s 앞자리가 ", " 인데 뒷자리는 나중에 알려드릴게요."),
    ("%s가 ", " 까지는 기억나는데 그 뒤가 생각이 안 나요."),
    ("일단 %s는 ", " 이렇게 적어주시고요."),
    ("%s는 ", " 로 시작하는 번호 맞습니다."),
    ("%s가 ", " 여기까지밖에 안 보여요."),
    ("%s 부분이 ", " 그다음은 흐려서 안 읽힙니다."),
    ("제가 아는 %s는 ", " 까지예요."),
]
P2_TPL = [
    ("%s가요 ", " 이에요."),
    ("아 그 %s는 ", " 뭐 이런 식이었어요."),
    ("%s가 ", " 맞을 거예요 아마."),
    ("어 그러니까 %s가 ", " 요."),
    ("%s 말씀이시죠, ", " 이거요."),
    ("%s는 ", " 이렇게 불러드리면 되나요?"),
]
P3_TPL = [
    ("%s는 ", " 입니다."),
    ("%s ", " 로 등록해 주세요."),
    ("%s가 ", " 이거 맞는지 확인 부탁드립니다."),
    ("작성하신 %s ", " 확인했습니다."),
    ("%s ", " 로 접수되었습니다."),
]

CAT_SPOKEN = {
    "주민등록번호": "주민번호", "외국인등록번호": "외국인등록번호",
    "여권번호": "여권번호", "운전면허번호": "운전면허번호",
    "계좌번호": "계좌번호", "카드번호": "카드번호", "연락처": "전화번호",
    "이메일 주소": "이메일", "국문 성명": "성함", "영문 성명": "영문 이름",
    "주소": "주소",
}


def make_record(idx, ptype, cat, full_value, rng, corp_to_opf):
    if ptype == "P1":
        surface = truncate_value(full_value, cat, rng)
        truncated = surface != full_value
        pre_t, suf = rng.choice(P1_TPL)
    elif ptype == "P2":
        if rng.random() < 0.3:
            surface = colloquialize(truncate_value(full_value, cat, rng), rng)
            truncated = True
        else:
            surface = colloquialize(full_value, rng)
            truncated = False
        pre_t, suf = rng.choice(P2_TPL)
    else:
        surface = respace(full_value, rng)
        truncated = False
        pre_t, suf = rng.choice(P3_TPL)

    prefix = pre_t % CAT_SPOKEN[cat]
    text = prefix + surface + suf
    start, end = len(prefix), len(prefix) + len(surface)
    assert text[start:end] == surface
    return {
        "text": text,
        "_surface": surface,
        "_start": start,
        "_end": end,
        "info": {
            "id": "SS-KO-PROBE-%05d" % idx,
            "probe_type": ptype,
            "corp_category": cat,
            "full_value": full_value,
            "truncated": truncated,
            "n_spans": 1,
            "char_len": len(text),
        },
        "_opf": corp_to_opf[cat],
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description="부분 발화 프로브셋 300건 생성 (P1/P2/P3 각 100건)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=os.path.join(root, "data",
                                                   "ss_pii_testset_ko_v1.json"),
                    help="정본 JSON 경로 (숫자형 가명값 재사용, 읽기 전용)")
    ap.add_argument("--pool", default=os.path.join(
        os.path.dirname(root), "과제1-2", "pseudonym_pool.json"),
        help="가명값 풀 경로 (성명/주소)")
    ap.add_argument("--out-dir", default=os.path.join(root, "data"),
                    help="출력 디렉토리")
    args = ap.parse_args()

    gold = load_json(args.gold, "정본 JSON")
    pool = load_json(args.pool, "가명값 풀")
    docs = gold.get("documents") if isinstance(gold, dict) else gold
    if not isinstance(docs, list):
        die("정본 JSON 에서 documents 배열을 찾을 수 없습니다: %s" % args.gold)

    corp_to_opf = {}
    for doc in docs:
        for sp in doc.get("spans", []):
            c, o = sp.get("corp_category"), sp.get("opf_label")
            if c and o:
                corp_to_opf.setdefault(c, o)
    for c in POOL_CATS | NUMERIC_CATS | {"이메일 주소"}:
        if c not in corp_to_opf:
            die("정본에서 '%s' 의 opf_label 매핑을 찾을 수 없습니다" % c)

    rng = random.Random(SEED)
    banks, residual = build_value_banks(docs, pool, rng)

    records, idx = [], 1
    cursor = {}
    regen = {}
    for ptype in ("P1", "P2", "P3"):
        for cat, n in ALLOC[ptype]:
            bank = banks.get(cat) or []
            if not bank:
                die("'%s' 항목의 가명값을 확보하지 못했습니다" % cat)
            for _ in range(n):
                # 절단·재간격으로 자릿수가 바뀌면 부분 문자열이 우연히
                # 검증식을 통과할 수 있다. 최종 발화 표면값 기준으로
                # 무효화를 강제하고, 실패하면 절단 지점과 값을 바꿔 재생성한다.
                attempts = 0
                while True:
                    k = cursor.get(cat, 0)
                    full = bank[k % len(bank)]
                    rec = make_record(idx, ptype, cat, full, rng, corp_to_opf)
                    if (value_is_invalidated(cat, rec["_surface"])
                            and surface_ok(cat, rec["_surface"])):
                        cursor[cat] = k + 1
                        break
                    attempts += 1
                    regen[cat] = regen.get(cat, 0) + 1
                    if attempts % 8 == 0:
                        cursor[cat] = k + 1      # 값 자체를 교체
                    if attempts > 200:
                        die("'%s' 항목에서 쓸 만한 표면값을 만들지 "
                            "못했습니다" % cat)
                records.append(rec)
                idx += 1

    if not os.path.isdir(args.out_dir):
        die("출력 디렉토리가 없습니다: %s" % args.out_dir)
    opf_path = os.path.join(args.out_dir, "probe_partial_utterance.jsonl")
    corp_path = os.path.join(args.out_dir, "probe_partial_utterance_corp.jsonl")

    with open(opf_path, "w", encoding="utf-8") as fo, \
            open(corp_path, "w", encoding="utf-8") as fc:
        for r in records:
            key_o = "%s: %s" % (r["_opf"], r["_surface"])
            key_c = "%s: %s" % (r["info"]["corp_category"], r["_surface"])
            off = [[r["_start"], r["_end"]]]
            fo.write(json.dumps({"text": r["text"], "spans": {key_o: off},
                                 "info": r["info"]}, ensure_ascii=False) + "\n")
            fc.write(json.dumps({"text": r["text"], "spans": {key_c: off},
                                 "info": r["info"]}, ensure_ascii=False) + "\n")

    # ---- 생성 후 자체 검증 ----
    by_type = {}
    off_bad = 0
    rrn_pass = frn_pass = luhn_pass = 0
    rrn_n = frn_n = card_n = 0
    trunc = 0
    for r in records:
        by_type[r["info"]["probe_type"]] = by_type.get(
            r["info"]["probe_type"], 0) + 1
        if r["text"][r["_start"]:r["_end"]] != r["_surface"]:
            off_bad += 1
        cat, sfc = r["info"]["corp_category"], r["_surface"]
        if r["info"]["truncated"]:
            trunc += 1
        if cat == "주민등록번호":
            rrn_n += 1
            rrn_pass += 1 if rrn_check_ok(sfc) else 0
        elif cat == "외국인등록번호":
            frn_n += 1
            frn_pass += 1 if (rrn_check_ok(sfc) or frn_check_ok(sfc)) else 0
        elif cat == "카드번호":
            card_n += 1
            luhn_pass += 1 if luhn_ok(sfc) else 0

    print("")
    print("=" * 74)
    print("프로브셋 생성 완료 (seed=%d)" % SEED)
    print("=" * 74)
    print("출력 (OPF 라벨판) : %s" % opf_path)
    print("출력 (사내 라벨판): %s" % corp_path)
    print("총 레코드          : %d" % len(records))
    print("유형별             : P1 %d / P2 %d / P3 %d" % (
        by_type.get("P1", 0), by_type.get("P2", 0), by_type.get("P3", 0)))
    print("절단(truncated)    : %d/%d" % (trunc, len(records)))
    print("오프셋 불일치      : %d/%d" % (off_bad, len(records)))
    print("")
    print("검증번호 통과 (0이어야 정상)")
    print("  주민등록번호     %d/%d" % (rrn_pass, rrn_n))
    print("  외국인등록번호   %d/%d  (주민식·구 +2식 둘 다 검사)" % (frn_pass, frn_n))
    print("  카드번호 Luhn    %d/%d" % (luhn_pass, card_n))
    print("")
    print("정본 값 중 검증식을 통과하는 잔여 건수 (0이어야 정상, 표본 제외는 하지 않음)")
    for cat in sorted(residual):
        if residual[cat]:
            print("  %-14s %d건  *** 회귀: 정본 무효화 누락 ***" % (cat, residual[cat]))
    if not any(residual.values()):
        print("  없음 (정본이 두 검증식 모두에서 무효화됨)")
    print("")
    print("표면값 재생성 횟수 (검증식 통과 또는 자릿수 부족)")
    if any(regen.values()):
        for cat in sorted(regen):
            print("  %-14s %d회" % (cat, regen[cat]))
    else:
        print("  없음")
    print("")


if __name__ == "__main__":
    main()
