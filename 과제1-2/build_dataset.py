#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Samsung Securities AI솔루션팀 OJT — 한국어 PII 마스킹 테스트셋 구축
Step 2: 한국어 전환 및 변환 (라벨 3단 매핑 / 스팬 병합 / 난이도 케이스 주입)
Step 3: 가명처리 (형식 유지 · 값 무효화 · 세트 내 일관성)
"""
import json, random, re, hashlib, os
from pathlib import Path
from collections import Counter, defaultdict

random.seed(20260814)
BASE = Path(__file__).resolve().parent
SRC = Path(os.environ.get('SS_SRC', BASE.parent / 'openpii_ko_sample.jsonl'))
OUT = Path(os.environ.get('SS_OUT', BASE))
OUT.mkdir(parents=True, exist_ok=True)
POOL_PATH = Path(os.environ.get('SS_POOL', BASE / 'pseudonym_pool.json'))

# ────────────────────────────────────────────────────────────
# 가명 값 풀 (Step 3)
#   성명 풀: pseudonym_pool.json 에서 로드한다.
#   Nemotron-Personas-Korea 반입 시 이 파일만 교체하면 코드 수정 없이 반영된다.
# ────────────────────────────────────────────────────────────
POOL = json.loads(POOL_PATH.read_text(encoding='utf-8'))
SURNAMES  = POOL['surnames']
SURNAME_W = POOL['surname_weights']
GIVEN     = POOL['given']
ROMAN_S   = POOL['romanization']['surname']
ROMAN_G   = POOL['romanization']['given']
assert len(SURNAMES) == len(SURNAME_W), 'surnames/surname_weights 길이 불일치'
_miss_s = [s for s in SURNAMES if s not in ROMAN_S]
_miss_g = [g for g in GIVEN if g not in ROMAN_G]
assert not _miss_s, f'romanization.surname 누락: {_miss_s[:10]}'
assert not _miss_g, f'romanization.given 누락: {_miss_g[:10]}'

SIDO = ['서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시',
        '울산광역시','세종특별자치시','경기도','강원특별자치도','충청북도','충청남도']
GUGUN = {'서울특별시':['중구','종로구','영등포구','강남구','서초구','마포구','송파구'],
         '부산광역시':['해운대구','부산진구','남구','수영구'],
         '대구광역시':['중구','수성구','달서구'],'인천광역시':['연수구','남동구','미추홀구'],
         '광주광역시':['서구','북구'],'대전광역시':['서구','유성구'],'울산광역시':['남구','중구'],
         '세종특별자치시':['한솔동','도담동'],
         '경기도':['수원시 영통구','성남시 분당구','고양시 일산동구','용인시 기흥구','화성시'],
         '강원특별자치도':['춘천시','원주시'],'충청북도':['청주시 흥덕구'],'충청남도':['천안시 서북구']}
ROAD = ['세종대로','테헤란로','여의대로','올림픽로','한강대로','반포대로','디지털로',
        '중앙로','산업로','효령로','불정로','광교중앙로','시청로','대학로']
EMAIL_DOM = ['naver.com','gmail.com','daum.net','hanmail.net','kakao.com','outlook.com']

# ────────────────────────────────────────────────────────────
# Step 2 — 라벨 3단 매핑 (openpii 36 → 사내 11 → OPF 8)
# ────────────────────────────────────────────────────────────
CORP_GROUP = {
    '주민등록번호':'고유식별정보','외국인등록번호':'고유식별정보',
    '여권번호':'고유식별정보','운전면허번호':'고유식별정보',
    '국문 성명':'개인식별정보','영문 성명':'개인식별정보','연락처':'개인식별정보',
    '계좌번호':'개인식별정보','이메일 주소':'개인식별정보','주소':'개인식별정보',
    '카드번호':'개인식별정보',
}
CORP2OPF = {
    '주민등록번호':'account_number','외국인등록번호':'account_number',
    '여권번호':'account_number','운전면허번호':'account_number',
    '계좌번호':'account_number','카드번호':'account_number',
    '국문 성명':'private_person','영문 성명':'private_person',
    '연락처':'private_phone','이메일 주소':'private_email','주소':'private_address',
}
# openpii 라벨 → 사내 항목
OPENPII2CORP = {
    'TAXNUM':'주민등록번호',        # 실측 4,962/4,962 가 RRN 정규식 일치
    'IDCARDNUM':'주민등록번호',      # 캐리어 문구가 이미 '주민등록번호/신분증 번호'
    'SOCIALNUM':'외국인등록번호',    # 국가발급 개인식별번호 슬롯을 국내 대응 항목으로 재배정
    'PASSPORTNUM':'여권번호',
    'DRIVERLICENSENUM':'운전면허번호',
    'TELEPHONENUM':'연락처',
    'EMAIL':'이메일 주소',
    'CREDITCARDNUMBER':'카드번호',
    'ACCOUNTNUM':'계좌번호',
}
NAME_LABELS = {'TITLE','GIVENNAME','SURNAME'}
ADDR_LABELS = {'CITY','STREET','BUILDINGNUM','ZIPCODE'}
# 비대상(과탐지 관찰용): DATE, AGE, SEX, GENDER, TIME, IDCARDNUM, URL, ORGANISATION 등

# ────────────────────────────────────────────────────────────
# Step 3 — 가명 값 생성기 (형식 유지 + 값 무효화)
# ────────────────────────────────────────────────────────────
def rrn_invalid(foreign=False):
    """주민등록번호/외국인등록번호 형식. 검증번호를 의도적으로 불일치시킴."""
    yy, mm, dd = random.randint(60,99), random.randint(1,12), random.randint(1,28)
    front = f'{yy:02d}{mm:02d}{dd:02d}'
    g = random.choice([5,6,7,8]) if foreign else random.choice([1,2,3,4])
    body = f'{g}{random.randint(0,99999):05d}'
    d = [int(c) for c in front+body]
    w = [2,3,4,5,6,7,8,9,2,3,4,5]
    chk = (11 - sum(a*b for a,b in zip(d,w)) % 11) % 10
    bad = (chk + random.randint(1,9)) % 10          # 검증번호 무효화
    return f'{front}-{body}{bad}'

def passport_kr():
    return f'M{random.randint(10000000,99999999)}'

def driver_kr():
    return (f'{random.randint(11,28):02d}-{random.randint(10,99):02d}-'
            f'{random.randint(100000,999999)}-{random.randint(10,99):02d}')

def card_invalid():
    """16자리. Luhn 체크섬 불일치."""
    pre = random.choice(['4','5'])
    body = pre + ''.join(str(random.randint(0,9)) for _ in range(14))
    s, alt = 0, True
    for c in reversed(body):
        n = int(c)*2 if alt else int(c)
        s += n-9 if n > 9 else n
        alt = not alt
    good = (10 - s % 10) % 10
    bad = (good + random.randint(1,9)) % 10
    n = body + str(bad)
    return f'{n[:4]}-{n[4:8]}-{n[8:12]}-{n[12:]}'

def account_kr():
    return f'{random.randint(100,999)}-{random.randint(10,99):02d}-{random.randint(100000,999999)}-{random.randint(10,99):02d}'

def phone_kr():
    return random.choice([
        f'010-{random.randint(1000,9999)}-{random.randint(1000,9999)}',
        f'010{random.randint(1000,9999)}{random.randint(1000,9999)}',
        f'02-{random.randint(200,999)}-{random.randint(1000,9999)}',
    ])

def address_kr():
    sido = random.choice(SIDO)
    return (f'{sido} {random.choice(GUGUN[sido])} {random.choice(ROAD)} '
            f'{random.randint(1,300)}')

def pick_name():
    s = random.choices(SURNAMES, weights=SURNAME_W, k=1)[0]
    g = random.choice(GIVEN)
    return s, g

# ────────────────────────────────────────────────────────────
# 스팬 병합 (Step 2): 인접 성명 / 인접 주소
# ────────────────────────────────────────────────────────────
def merge_runs(text, spans, labelset, max_gap=4):
    out, i = [], 0
    spans = sorted(spans, key=lambda s: s['start'])
    while i < len(spans):
        if spans[i]['label'] not in labelset:
            out.append(spans[i]); i += 1; continue
        j = i
        while (j+1 < len(spans) and spans[j+1]['label'] in labelset
               and spans[j+1]['start'] - spans[j]['end'] <= max_gap
               and re.fullmatch(r'[\s,·]*', text[spans[j]['end']:spans[j+1]['start']] or '')):
            j += 1
        if j > i:
            out.append({'label':'__MERGED__','members':[s['label'] for s in spans[i:j+1]],
                        'start':spans[i]['start'],'end':spans[j]['end'],
                        'value':text[spans[i]['start']:spans[j]['end']]})
        else:
            out.append(spans[i])
        i = j+1
    return out

# ────────────────────────────────────────────────────────────
# Step 2-1: 캐리어 문구 한국어 정규화 (문맥 단서 ↔ 값 형식 ↔ 라벨 일치)
CARRIER = [
    ('세금 식별 번호','주민등록번호'), ('세금 식별번호','주민등록번호'),
    ('세금식별번호','주민등록번호'), ('납세자 번호','주민등록번호'),
    ('사회보장 번호','외국인등록번호'), ('사회보장번호','외국인등록번호'),
    ('Social Security No.','외국인등록번호'), ('신분증 번호','주민등록번호'),
    ('National ID','주민등록번호'),
]

def normalize_carrier(text, spans):
    """캐리어 문구를 치환하고 스팬 오프셋을 함께 이동시킨다."""
    for old, new in CARRIER:
        while True:
            i = text.find(old)
            if i < 0:
                break
            if any(s['start'] < i + len(old) and i < s['end'] for s in spans):
                break                      # 스팬과 겹치면 건드리지 않음
            delta = len(new) - len(old)
            text = text[:i] + new + text[i+len(old):]
            for s in spans:
                if s['start'] >= i + len(old):
                    s['start'] += delta; s['end'] += delta
            if delta == 0 and old == new:
                break
    return text, spans


def process(row, doc_idx):
    text = row['source_text']
    spans = [dict(label=p['label'], start=p['start'], end=p['end'], value=p['value'])
             for p in row['privacy_mask']]
    spans = [s for s in spans if text[s['start']:s['end']] == s['value']]
    text, spans = normalize_carrier(text, spans)
    spans = [s for s in spans if text[s['start']:s['end']] == s['value']]
    spans = merge_runs(text, spans, NAME_LABELS)
    spans = merge_runs(text, spans, ADDR_LABELS)
    spans.sort(key=lambda s: s['start'])

    # 문서 내 일관성 키: 같은 원본 값 → 같은 가명 값
    consistent = {}
    def keep(kind, orig, gen):
        k = (kind, orig)
        if k not in consistent:
            consistent[k] = gen()
        return consistent[k]

    new, cur, out_spans = [], 0, []
    for s in spans:
        if s['start'] < cur:            # 중첩 스팬 방어
            continue
        new.append(text[cur:s['start']])
        lab = s['label']
        corp = None; val = None; pseudo = True

        if lab == '__MERGED__':
            mem = set(s.get('members', []))
            if mem & ADDR_LABELS:
                corp = '주소'; val = keep('addr', s['value'], address_kr)
            else:
                latin = bool(re.search(r'[A-Za-z]', s['value']))
                sur, giv = keep('name', s['value'], pick_name)
                if latin:
                    corp = '영문 성명'; val = f'{ROMAN_G[giv]} {ROMAN_S[sur]}'
                else:
                    corp = '국문 성명'; val = f'{sur}{giv}'
        elif lab in ADDR_LABELS:
            corp = '주소'; val = keep('addr', s['value'], address_kr)
        elif lab in NAME_LABELS:
            sur, giv = keep('name', s['value'], pick_name)
            if re.search(r'[A-Za-z]', s['value']):
                corp = '영문 성명'; val = f'{ROMAN_G[giv]} {ROMAN_S[sur]}'
            else:
                corp = '국문 성명'; val = f'{sur}{giv}'
        elif lab in ('TAXNUM', 'IDCARDNUM'):
            corp = '주민등록번호'; val = keep('rrn', s['value'], lambda: rrn_invalid(False))
        elif lab == 'SOCIALNUM':
            corp = '외국인등록번호'; val = keep('arn', s['value'], lambda: rrn_invalid(True))
        elif lab == 'PASSPORTNUM':
            corp = '여권번호'; val = keep('pp', s['value'], passport_kr)
        elif lab == 'DRIVERLICENSENUM':
            corp = '운전면허번호'; val = keep('dl', s['value'], driver_kr)
        elif lab == 'TELEPHONENUM':
            corp = '연락처'; val = keep('tel', s['value'], phone_kr)
        elif lab == 'CREDITCARDNUMBER':
            corp = '카드번호'; val = keep('card', s['value'], card_invalid)
        elif lab == 'ACCOUNTNUM':
            corp = '계좌번호'; val = keep('acct', s['value'], account_kr)
        elif lab == 'EMAIL':
            def mkmail():
                sur, giv = pick_name()
                return f'{ROMAN_G[giv].lower()}.{ROMAN_S[sur].lower()}{random.randint(10,99)}@{random.choice(EMAIL_DOM)}'
            corp = '이메일 주소'; val = keep('mail', s['value'], mkmail)
        else:
            # 비대상 라벨: 원문 유지, 과탐지 관찰 대상
            new.append(s['value']); cur = s['end']; continue

        st = sum(len(x) for x in new)
        new.append(val)
        out_spans.append({'start':st, 'end':st+len(val), 'value':val,
                          'source_label':('+'.join(s.get('members',[])) if lab=='__MERGED__' else lab),
                          'corp_category':corp, 'corp_group':CORP_GROUP[corp],
                          'opf_label':CORP2OPF[corp], 'pseudonymized':pseudo,
                          'injected':False})
        cur = s['end']
    new.append(text[cur:])
    ntext = ''.join(new)

    # ── Step 2-5: 난이도 케이스 주입 ──────────────────────────
    diff = []
    fin = any(k in ntext for k in ['결제','카드','계좌','투자','세금','은행','영수증','회의록','이체','송금'])
    have = {sp['corp_category'] for sp in out_spans}

    def append_sentence(prefix, value, corp, tail):
        nonlocal ntext
        base = len(ntext) + len(prefix)
        ntext = ntext + prefix + value + tail
        out_spans.append({'start':base,'end':base+len(value),'value':value,
                          'source_label':'INJECTED','corp_category':corp,
                          'corp_group':CORP_GROUP[corp],'opf_label':CORP2OPF[corp],
                          'pseudonymized':True,'injected':True})

    # (1) 계좌번호 — 원본 12건뿐이라 금융 문맥 행에 주입
    if fin and '계좌번호' not in have and doc_idx % 2 == 0:
        append_sentence('\n증권계좌번호는 ', account_kr(), '계좌번호', '이며, 출금은 본인 명의 계좌로만 가능합니다.')
        diff += ['조사결합']
    # (2) 외국인등록번호 보강
    if '외국인등록번호' not in have and doc_idx % 7 == 0:
        append_sentence('\n외국인 투자자의 외국인등록번호는 ', rrn_invalid(True), '외국인등록번호', '으로 확인되었습니다.')
        diff += ['조사결합']
    # (3) 경칭 결합 + 한영 혼용
    if doc_idx % 3 == 0:
        sur, giv = pick_name()
        append_sentence('\n담당 PB ', f'{sur}{giv}', '국문 성명', f' 수석({ROMAN_G[giv]} {ROMAN_S[sur]})이 응대하였습니다.')
        diff += ['경칭결합','한영혼용']
    # (4) 고엔트로피 비PII 과탐지 유도 — 종목코드·주문번호 (정답 스팬 없음)
    if fin and doc_idx % 4 == 0:
        ntext += (f'\n주문번호 {random.randint(20260101,20260831)}-{random.randint(100000,999999)},'
                  f' 종목코드 {random.choice(["005930","000660","035420","207940","051910"])} 건은 정상 체결되었습니다.')
        diff += ['과탐지유도']

    out_spans.sort(key=lambda s: s['start'])
    assert all(ntext[s['start']:s['end']] == s['value'] for s in out_spans)

    masked = []
    cur = 0
    for s in out_spans:
        masked.append(ntext[cur:s['start']]); masked.append(f"[{s['opf_label']}]"); cur = s['end']
    masked.append(ntext[cur:])

    return {
        'id': f'SS-KO-{doc_idx+1:05d}',
        'source_uid': row['uid'],
        'split': row['split'],
        'text': ntext,
        'masked_text': ''.join(masked),
        'spans': out_spans,
        'meta': {
            'n_spans': len(out_spans),
            'n_injected': sum(1 for s in out_spans if s['injected']),
            'finance_context': fin,
            'difficulty_cases': sorted(set(diff)),
            'char_len': len(ntext),
            'word_len': len(ntext.split()),
        }
    }

# ────────────────────────────────────────────────────────────
rows = [json.loads(l) for l in SRC.open(encoding='utf-8')]
docs = [process(r, i) for i, r in enumerate(rows)]

# 산출물 1 — 정본 (사내 11항목 3단 매핑 + 가명처리 완료)
with (OUT/'ss_pii_testset_ko_v1.json').open('w', encoding='utf-8') as f:
    json.dump({'dataset':'Samsung Securities Korean PII Test Set',
               'version':'v1','built':'2026-08-14',
               'source':{'name':'ai4privacy/pii-masking-openpii-1.5m',
                         'license':'CC BY 4.0','subset':'language=ko, region=KR',
                         'sampled':len(rows)},
               'label_scheme':{'corp_categories':list(CORP_GROUP.keys()),
                               'opf_labels':sorted(set(CORP2OPF.values())),
                               'openpii_to_corp':OPENPII2CORP},
               'pseudonymization':{'guideline':'개인정보보호위원회 「가명정보 처리 가이드라인」 2026.03.',
                                   'principles':['형식 유지','검증번호·Luhn 불일치로 값 무효화',
                                                 '문서 내 일관 치환','범주화 미적용'],
                                   'name_pool_source':POOL.get('note','')},
               'documents':docs}, f, ensure_ascii=False, indent=1)

# 산출물 2 — OPF 평가 입력 포맷 (jsonl)
with (OUT/'ss_pii_testset_ko_v1_opf.jsonl').open('w', encoding='utf-8') as f:
    for d in docs:
        f.write(json.dumps({'id':d['id'],'text':d['text'],
                            'spans':[{'start':s['start'],'end':s['end'],
                                      'label':s['opf_label'],'text':s['value']}
                                     for s in d['spans']]}, ensure_ascii=False)+'\n')

# 산출물 3 — 가명 값 풀 (교체 가능)
#   입력 풀을 그대로 라운드트립한다. 주소/이메일 도메인은 코드 상수를 정본으로 동기화.
_pool_out = dict(POOL)
_pool_out['address'] = {'sido':SIDO,'gugun':GUGUN,'road':ROAD}
_pool_out['email_domains'] = EMAIL_DOM
with (OUT/'pseudonym_pool.json').open('w', encoding='utf-8') as f:
    json.dump(_pool_out, f, ensure_ascii=False, indent=1)

# ── 검증 리포트 ──
cat = Counter(s['corp_category'] for d in docs for s in d['spans'])
grp = Counter(s['corp_group'] for d in docs for s in d['spans'])
opf = Counter(s['opf_label'] for d in docs for s in d['spans'])
inj = Counter(s['corp_category'] for d in docs for s in d['spans'] if s['injected'])
tot = sum(cat.values())
print(f'문서 {len(docs):,} / 스팬 {tot:,} / 주입 {sum(inj.values()):,}')
print('\n[사내 11개 항목별 빈도]')
for k in CORP_GROUP:
    print(f'  {CORP_GROUP[k]:<8}{k:<12}{cat[k]:>7,}  (주입 {inj[k]:,})')
print('\n[OPF 8라벨 환산]')
for k,v in opf.most_common(): print(f'  {k:<18}{v:>7,}')
print('\n[난이도 케이스]')
dc = Counter(c for d in docs for c in d['meta']['difficulty_cases'])
for k,v in dc.most_common(): print(f'  {k:<12}{v:>6,} 문서')
print(f"\n금융 문맥 문서 {sum(1 for d in docs if d['meta']['finance_context']):,}/{len(docs):,}")
import statistics
print('평균 문자 %.1f / 평균 어절 %.1f'%(
    statistics.mean(d['meta']['char_len'] for d in docs),
    statistics.mean(d['meta']['word_len'] for d in docs)))
