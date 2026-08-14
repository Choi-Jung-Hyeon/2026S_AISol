# 1주차 OJT — OPF 검증용 한국어 PII 테스트셋 구축 (인수인계 문서)

작성 2026.08.14 / 최중현 (AI솔루션팀 OJT)

---

## 1. 한 줄 요약

공개 데이터셋 `ai4privacy/pii-masking-openpii-1.5m`의 한국어 서브셋을 사내 11개 마스킹 항목 체계로
재매핑하고, 「가명정보 처리 가이드라인」 원칙에 따라 전 PII 값을 한국 실물 형식의 가명값으로
교체한 OPF 평가용 테스트셋입니다. 문서 3,000건 / PII 스팬 28,420건.

가명 성명 풀은 `nvidia/Nemotron-Personas-Korea` (CC BY 4.0) 5만 건 실측 분포로 교체 완료
(성씨 40종 · 이름 300종). 근사 풀 사용 단계는 종료되었습니다.

---

## 2. 과제 원 지시 대비 변경 사항 (보고 필요)

| 항목 | 원 지시 | 실제 수행 | 사유 |
| --- | --- | --- | --- |
| 데이터셋 | PII-Masking-300k | pii-masking-openpii-1.5m | 300k는 학술·비상업 전용 라이선스로 상업적 파생물 생성 불가. 동일 제작기관(ai4privacy)의 CC BY 4.0 최신 플래그십으로 교체 |
| 가이드라인 판본 | 2024.2 | 2026.3 | 2026.3 전면 개정으로 위험도 판단 기준 표준화, 서식 24종→10종 축소. 2024.2 기준 일부 용어가 현행본에 부재 |
| 성명 로마자 표기 | Nemotron 실측 매핑으로 교체 | 관용 표기 39개 유지 + 규칙 전자 301개 보충 | Nemotron 26개 컬럼에 로마자 표기 컬럼이 **부재**(README 명시: "이름·성 … 필드들은 포함하지 않습니다"). 실측 교체 불가하여 기존 관용 표기(KIM/LEE/PARK…)는 보존하고, 미커버 성명만 「국어의 로마자 표기법」 제3장 제4항(인명은 음운 변화를 표기에 반영하지 않음)에 따라 음절 단위 전자로 보충 |

세 건 모두 임 수석님께 사전 구두 보고 필요.

로마자 건은 항목별 출처가 `pseudonym_pool.json`의 `romanization_provenance` 키에 기록되어 있어,
실측 유지분과 규칙 생성분을 언제든 구분·감사할 수 있습니다.

---

## 3. 파일 구성

### 3-1. 최종 산출물

| 파일 | 용도 |
| --- | --- |
| `1주차_OPF_한국어_PII_테스트셋_구축.pptx` | 발표 자료 12장 |
| `ss_pii_testset_ko_v1.json` | **정본 데이터셋**. 문서별 본문·마스킹본·스팬(3단 라벨) |
| `ss_pii_testset_ko_v1_opf.jsonl` | OPF 평가 입력 포맷 |
| `pseudonym_pool.json` | 가명 값 풀. **`build_dataset.py`가 런타임에 읽는 입력** (Nemotron 실측 반영 완료) |
| `build_dataset.py` | 재현 스크립트 |
| `openpii_ko_label_dictionary.json` | openpii 36개 라벨 정의표 (빈도·값 패턴·사내 항목 대응). 원문 PII 미포함 |

### 3-2. 중간 산출물 (재현·검증용)

| 파일 | 용도 |
| --- | --- |
| `nemotron_names.json` | Nemotron 5만 건에서 추출한 성명·지역 원자료 (고유 성명 30,612종) |
| `openpii_ko_sample.jsonl` | 원본 표집분 3,000건. `build_dataset.py`의 입력 |
| `openpii_ko_labels.json` | 라벨 빈도·값 패턴 요약 |
| `openpii_ko_full.jsonl` | ko 전체 26,498행 (용량 큼, 선택) |

> `openpii_ko_*.jsonl` 3종은 용량 문제로 zip에 포함하지 않았습니다. 상위 폴더에 있습니다.

---

## 4. 정본 데이터셋 스키마

```jsonc
{
  "dataset": "Samsung Securities Korean PII Test Set",
  "version": "v1",
  "source": { "name": "...", "license": "CC BY 4.0", "subset": "language=ko, region=KR" },
  "label_scheme": { "corp_categories": [...11개...], "opf_labels": [...], "openpii_to_corp": {...} },
  "pseudonymization": { "guideline": "...", "principles": [...], "name_pool_source": "..." },
  "documents": [
    {
      "id": "SS-KO-00001",
      "source_uid": 24644953,          // 원본 openpii uid — 역추적용
      "split": "train",
      "text": "가명처리 완료된 한국어 본문",
      "masked_text": "[account_number] 형태로 치환된 정답 마스킹본",
      "spans": [
        {
          "start": 8, "end": 18, "value": "GEUMSUK YU",
          "source_label": "TITLE+GIVENNAME+SURNAME", // 원본 openpii 라벨 (병합 시 + 로 연결)
          "corp_category": "영문 성명",           // 사내 11개 항목
          "corp_group": "개인식별정보",           // 고유식별정보 / 개인식별정보
          "opf_label": "private_person",         // OPF 8라벨
          "pseudonymized": true,
          "injected": false                      // true = Step 2에서 신규 주입한 스팬
        }
      ],
      "meta": {
        "n_spans": 14, "n_injected": 2,
        "finance_context": true,
        "difficulty_cases": ["경칭결합", "과탐지유도", "조사결합", "한영혼용"],
        "char_len": 1003, "word_len": 174
      }
    }
  ]
}
```

### 평가 시 주의

- `injected: true` 스팬은 원본 유래분과 **분리 집계**할 것. 합산하면 원본 데이터셋 성능이 아님
- `opf_label` 기준 집계와 `corp_category` 기준 집계를 **모두** 산출할 것.
  OPF의 `account_number` 하나에 사내 6개 항목이 수렴하므로, OPF 라벨 기준만으로는
  주민등록번호와 카드번호를 구분할 수 없음
- `difficulty_cases` 가 비어 있지 않은 문서는 별도 서브셋으로 분리 측정 권장
- 과탐지 유도 문자열(종목코드·주문번호)은 **정답 스팬이 없음**. 모델이 잡으면 FP

---

## 5. 재현 절차

```bash
# 입력: openpii_ko_sample.jsonl (상위 폴더), pseudonym_pool.json (같은 폴더)
# 출력: ss_pii_testset_ko_v1.json, ss_pii_testset_ko_v1_opf.jsonl, pseudonym_pool.json
python3 build_dataset.py
```

- 시드 고정(`random.seed(20260814)`)이므로 동일 입력에 대해 동일 산출물이 재현됨
- **가명 값 풀만 바꾸려면 `pseudonym_pool.json`만 교체하면 됩니다. 스크립트는 손대지 않습니다.**
  성명 풀은 하드코딩되어 있지 않고 런타임에 이 파일에서 로드됩니다
  (`surnames` / `surname_weights` / `given` / `romanization.surname` / `romanization.given`)
- 풀 교체 시 `given`·`surnames`의 모든 항목이 `romanization`에 존재해야 합니다.
  누락되면 기동 시점의 `assert`가 누락 목록과 함께 즉시 실패합니다 (실행 중 `KeyError` 방지)
- 경로는 환경변수로 덮어쓸 수 있습니다 — `SS_SRC`(입력) / `SS_OUT`(출력) / `SS_POOL`(풀).
  미지정 시 스크립트 위치 기준 상대경로를 사용합니다

### 풀 교체 시 유지되어야 하는 값

성명 풀을 바꿔도 아래 수치는 **변하지 않아야 합니다**. 달라지면 변환 로직에 영향을 준 것이므로
원인을 먼저 확인하십시오. (Nemotron 반입 재실행에서 전 항목 일치 확인 완료)

문서 3,000 / 스팬 28,420 / 11개 항목별 빈도 / 오프셋 28,420건 전건 일치 /
주민·외국인등록번호 검증번호 통과 0건 / 카드 Luhn 통과 0건

---

## 6. 검증된 수치 (실측)

| 항목 | 값 |
| --- | --- |
| 문서 | 3,000 |
| PII 스팬 | 28,420 (주입 2,048 포함) |
| 사내 11개 항목 커버 | 11/11 |
| 오프셋 정합 (`text[start:end] == value`) | 28,420/28,420 |
| 주민·외국인등록번호 검증번호 통과 | 0/4,900 |
| 카드번호 Luhn 통과 | 0/1,788 |
| 금융 문맥 문서 | 1,569/3,000 |

마지막 두 줄이 Step 3의 핵심 근거 — 형식은 유효하되 실재할 수 없는 값임을 확인.

위 수치는 Nemotron 실측 풀 반입 후 재실행에서 전 항목 동일하게 재현되었습니다.
성명 값만 5,806/5,807건이 교체되었고, 스팬 구조·오프셋·무효화 검증은 불변입니다.

### 6-1. 원본 ko 서브셋 기준 통계 (전수 26,498행)

| 항목 | 값 |
| --- | --- |
| 전체 데이터셋 대비 한국어 | 26,498/1,636,375 (1.62%, 30개 언어 중 24위) |
| ko 조합 | `language=ko` · `region=KR` · `script=Hang` 단일 |
| PII 항목 총계 | 189,429 (행당 평균 7.15) |
| 오프셋 정합 | 189,429/189,429 |
| 라벨 종류 | 36종 (사내 11개 항목 대응 16종 / 미대응 20종) |
| 문서 길이 | 평균 181.3자 · 34.9어절 (중앙값 140자) |
| 문장 길이 | 평균 48.9자 · 9.6어절 (총 96,592문장, 문서당 3.65문장) |

---

## 7. 미완료 / 검증 필요 항목

| 항목 | 상태 |
| --- | --- |
| OPF 입력 분할 | **미결정.** 문서 평균 464자(가명처리 후) vs OPF 실효 참조 범위 257토큰. 문장 평균은 48.9자이므로 문장 단위 분할 시 컨텍스트 부족 우려 — 아래 참고 |

### 7-1. 해소된 항목 (2026.08.14 2차 작업)

| 항목 | 결과 |
| --- | --- |
| 언어별 분포 | 30개 언어 전량 산출. 한국어 26,498/1,636,375 |
| 문장 단위 평균 길이 | 96,592문장 / 평균 48.9자 · 9.6어절 |
| PII 라벨 36종 정의표 | `openpii_ko_label_dictionary.json` |
| 전체 26,498행 기준 통계 | §6-1 |
| Nemotron-Personas-Korea 실물 | 반입 완료. 성씨 상위 10위 중 8개가 기존 근사 풀과 순위 일치, 최대 편차 정(4.3→4.87) |
| 표집 편향 | 정량화 완료. 아래 참고 |

### 7-2. 표집 3,000건의 편향 (평가 해석 시 필수)

라벨 다양성 우선 표집이라 **행당 평균 PII가 전체 7.15 vs 표집 15.99로 2.2배**입니다.
희소 라벨 16종(`ACCOUNTNUM` `BANKNAME` `ORGANISATION` 등)은 원본이 50행 미만이라 전량 포함되어
항목 배율 3.95배인 반면, `GIVENNAME`(0.78) `TIME`(0.79) `DATE`(0.87)는 상대적으로 과소표집입니다.

→ 이 테스트셋의 재현율은 **다중 PII 밀집 문서에 편중된 값**입니다. 전체 모집단 성능으로
   일반화하지 마십시오.

### 7-3. 미대응 20종의 과탐지 영향

사내 11개 항목에 대응하지 않는 20개 라벨이 원본 PII의 42,992/189,429 (22.70%)를 차지합니다
(`DATE` 21,077 / `AGE` 9,733 / `GENDER` 5,433 / `SEX` 4,551 / `TIME` 1,964 등).
대응 16종은 146,437/189,429 (77.30%)입니다.
이들은 정답 스팬 없이 원문 그대로 남아 있으므로, OPF가 탐지하면 전부 FP로 계산됩니다.
의도된 설계이며, 과탐지 경향 측정에 사용하십시오.

---

## 8. 출처 표기 (CC BY 4.0 준수)

```
본 데이터셋은 ai4privacy/pii-masking-openpii-1.5m (Ai4Privacy, CC BY 4.0)의
한국어 서브셋을 가공하여 제작하였습니다.

가명처리 기준: 개인정보보호위원회, 「가명정보 처리 가이드라인」 2026.03.
```

발표 자료 및 산출물 배포 시 위 문구를 반드시 포함할 것.
