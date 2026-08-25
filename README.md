# 2026S AI Solution OJT — OPF 한국어 PII 평가

OpenAI Privacy Filter(OPF)를 사내 도입할 수 있는지 판단하기 위해,
한국어 PII 테스트셋을 만들고 그것으로 모델을 평가하는 과제입니다.

- **`dataset/`** — 평가할 데이터를 만든다 (1주차)
- **`eval/`** — 그 데이터로 모델을 평가한다 (2주차)

모델 체크포인트를 로컬에 반입해 **직접 추론이 가능합니다.** 가중치는
`~/.opf/privacy_filter` (리포 밖, 추적하지 않음)에 있고, `eval/opf_local.py` 가
추론 어댑터입니다.

---

## 디렉토리

```
dataset/    정본 테스트셋, 생성기, 가명값 풀, 원천 입력(source/)
eval/       평가 파이프라인 — 로컬 추론 어댑터·집계 스크립트·운영점·실행 절차
reference/  참고 자료 — OPF 공개 저장소 사본(opf/), KDPII 실물(kdpii/),
            모델카드(model-card/), tiktoken 캐시(tiktoken-cache/)
reports/    전달 문서 — 인수인계 2종, 문의회신·실행가이드, 발표자료(slides/)
archive/    미사용·폐기 예정 — 1주차 초안(week1-draft/), 반입 zip(packaging/)
```

### dataset/ (정본)

| 파일 | 내용 |
| --- | --- |
| `ss_pii_testset_ko_v1.json` | **정본.** 문서 3,000건 / PII 스팬 28,420건 |
| `ss_pii_testset_ko_v1_opf.jsonl` | 정본의 OPF 라벨 JSONL 판 |
| `eval_opf_labels.jsonl` | 하니스 스키마, OPF 5라벨 |
| `eval_corp_labels.jsonl` | 하니스 스키마, 사내 11항목 |
| `build_dataset.py` | 정본 생성기 (seed 20260814) |
| `convert_schema.py` | 정본 → 하니스 스키마 변환기 |
| `pseudonym_pool.json` | 가명값 풀 (성명·주소·이메일 도메인) |
| `nemotron_names.json` | Nemotron 5만 건 추출 성명 원자료 |
| `build_nemotron_names.py` | parquet → `nemotron_names.json` |
| `build_pseudonym_pool.py` | `nemotron_names.json` → 가명값 풀 |
| `source/` | openpii 원천 (`build_dataset.py` 의 기본 입력) |

1주차 상세 인수인계는 `reports/README_인수인계.md` 로 옮겼습니다.

### eval/ (평가)

```
opf_local.py     로컬 추론 어댑터 — str -> (spans, masked_text)
opf_probe.py     토큰·오프셋·예측태그 전수 덤프 (원인 규명용)
opf_predict.py   정본 전량 추론 -> 예측 스팬 JSONL
cache_logits.py  로짓 [T,33] + 오프셋 캐시 (추론 1회, 이후 재추론 없음)
viterbi.py       제약 Viterbi 디코더 (BIOES 전이 제약) — 정본 디코딩 경로
decode_compare.py argmax vs Viterbi 비교 + 축별 대표 케이스 선별
build_report_html.py 좌우 비교 HTML 보고서 생성 (자기완결형)
postproc.py      group_spans / mask_text (원본 전사, 수정 금지)
data/          정본·eval 라벨·스모크 미니셋·부분 발화 프로브셋
scripts/       집계 스크립트 9종
configs/       Viterbi 운영점 3종
results/       실행 결과 (예측 JSONL 등은 재생성 가능하므로 추적하지 않음)
RUNBOOK.txt    실행 순서 [0]~[7]
```

| 스크립트 | 하는 일 |
| --- | --- |
| `aggregate_recall.py` | 사내 11항목 재현율, OPF 5라벨 span 지표, 고유식별 4종 미탐 |
| `doc_level_miss.py` | 문서 단위 미탐율, 항목별 분류, 미탐 스팬 샘플 |
| `fp_breakdown.py` | 과탐 3분해 (B2 업무영향 / B1 설계상 / B3 순수) |
| `build_probe_set.py` | 부분 발화 프로브셋 300건 생성 (seed 20260818) |
| `rule_layer.py` | 고유식별정보 4종 정규식 탐지 |
| `hybrid_merge.py` | OPF 단독 / 규칙 단독 / 합집합 3단 비교 |
| `standalone_metrics.py` | 하니스 없이 예측 스팬만으로 전 지표 산출 |
| `dump_examples.py` | 미탐·과탐 실사례를 6버킷(A~F)으로 추출 |
| `build_review_sheet.py` | 사람 검수용 CSV 300건 표집 |

단일 진입점은 `eval/run_all.py` 이며 위 중 5종을 순서대로 호출합니다.
`run_opf.py`(개발망 전용)는 `eval/` 바로 아래에 놓으면 됩니다.

---

## 데이터셋 수치 (실측)

| 항목 | 값 |
| --- | --- |
| 문서 | 3,000 |
| PII 스팬 | 28,420 (주입 2,048 포함) |
| 오프셋 정합 `text[start:end] == value` | 28,420/28,420 |
| 사내 항목 커버 | 11/11 |
| 금융 문맥 문서 | 1,569/3,000 |

사내 11항목: 주소 6,423 / 주민등록번호 3,320 / 국문 성명 3,295 / 이메일 3,095 /
연락처 2,803 / 영문 성명 2,512 / 카드번호 1,788 / 외국인등록번호 1,580 /
운전면허번호 1,506 / 여권번호 1,283 / 계좌번호 815

OPF 5라벨: account_number 10,292 / private_address 6,423 / private_person 5,807 /
private_email 3,095 / private_phone 2,803

### 값 무효화

전량 합성 가명값이며, 형식은 유효하되 실재할 수 없도록 검증번호를 어긋냅니다.

| 검증식 | 통과 |
| --- | --- |
| 주민등록번호 | 0/3,320 |
| 외국인등록번호 (주민식·구 외국인식 양쪽) | 0/1,580 |
| 카드번호 Luhn | 0/1,788 |

구 외국인등록번호 검증식은 유효 검증번호가 주민식 `+2`라 별도 처리가 필요합니다.
초기 산출물은 주민식 기준으로만 무효화해 1,580건 중 166건이 구 검증식을
통과했고, 이후 보수적으로 양쪽 모두 회피하도록 수정했습니다.

---

## 평가 방침

**미탐(FN)이 가장 치명적입니다.** 우선순위는 Recall > F2 > F1 > Precision이며,
Accuracy는 배경 토큰 비중 때문에 과대평가되어 주지표에서 제외합니다.
과탐은 허용 범위이나 업무영향 과탐(주문번호·종목코드)은 별도 집계합니다.

### 스팬 매칭은 비대칭입니다

재현율 TP는 `정답 ⊆ 예측`, 정밀도 TP는 `예측 ⊆ 정답`으로 **두 TP 개수가 다릅니다.**
집계 스크립트는 두 축을 분리해 계산하며, FN과 FP를 합산한 값은 만들지 않습니다.

### 조인은 text 완전 일치로만 합니다

`opf eval --predictions-out`의 `example_id`는 sha256 자동 생성값이라 정본 id와
다릅니다. 모든 스크립트는 text 완전 일치로 조인하고, 실패 건수가 0이 아니면
집계를 중단합니다.

### 수치는 분수로 표기합니다

백분율을 쓰지 않습니다. recall/precision은 분수(`4/4900`), f1/f2는 조화평균이라
분수로 환원되지 않으므로 0~1 비율로 씁니다.

---

## 평가 결과 (정본 3,000건 전수)

### 지표 정의 — 3자 기준, 다대일

마스킹이 목적이므로 **라벨이 틀려도 문자만 덮이면 성공**입니다. 지표는 하나뿐입니다.

- 정답과 예측이 **3자 이상 겹치면 TP**. 정답이 3자 미만이면 전량 피복을 요구합니다.
- **다대일 허용** — 예측 하나가 여러 정답을 덮으면 그 정답을 전부 TP 로 인정합니다.
- **FP** = 어떤 정답과도 3자 이상 겹치지 않는 예측 스팬.
- **재현율이 주지표, F1 이 보조지표**입니다. 정밀도·F2 는 내지 않습니다.

### argmax vs 제약 Viterbi

| | argmax | **Viterbi (정본)** | 차이 |
| --- | --- | --- | --- |
| 재현율 | 0.9181 | 0.9166 | −0.0015 |
| F1 | 0.8213 | **0.8686** | **+0.0473** |
| TP / FN / FP | 26,093 / 2,327 / 9,028 | 26,050 / 2,370 / **5,509** | FP −3,519 |
| 예측 스팬 수 | 34,649 | 29,587 | −5,062 |
| **금지 전이** | **8,745** | **0** | −8,745 |
| 길이 1토큰 스팬 | 1,511 | 3 | −1,508 |

### 상태 3분류 (지표 아님 — 상태 분포)

| | argmax | Viterbi | 차이 |
| --- | --- | --- | --- |
| 완전 피복 | 25,564 | **25,859** | **+295** |
| 부분 노출 | 710 | **214** | **−496** |
| 완전 미탐 | 2,146 | 2,347 | +201 |
| 합 | 28,420 | 28,420 | 정답 스팬과 일치 |
| **부분 노출 문자 수** | 3,889 | **1,326** | **−2,563 (−66%)** |

**재현율과 상태 분포가 반대 방향을 가리킵니다.** 재현율은 −0.0015 로 사실상 동률이지만,
마스킹 실물로는 부분 노출이 496건, 노출 문자가 2,563자 줄었습니다. Viterbi 가 예측을
병합해 스팬 수가 줄어든 것이 재현율에 미세하게 불리하게 작용한 결과입니다.
**보고 시 두 표를 반드시 함께 실어야 합니다.**

### 사내 11항목별 재현율

| 항목 | argmax | Viterbi | ΔR |
| --- | --- | --- | --- |
| 이메일 주소 / 계좌번호 | 1.0000 | 1.0000 | ±0 |
| 외국인등록번호 | 0.9987 | 0.9981 | −0.0006 |
| 운전면허번호 | 0.9987 | 0.9973 | −0.0013 |
| 연락처 | 0.9979 | 0.9971 | −0.0007 |
| 주민등록번호 | 0.9973 | 0.9961 | −0.0012 |
| 카드번호 | 0.9944 | 0.9933 | −0.0011 |
| 여권번호 | 0.9930 | 0.9922 | −0.0008 |
| 주소 | 0.9355 | 0.9273 | −0.0083 |
| 영문 성명 | 0.7743 | 0.7675 | −0.0068 |
| **국문 성명** | 0.6030 | **0.6149** | **+0.0118** |

Viterbi 가 재현율이 낮아진 항목은 8개이며 전부 −0.01 미만입니다. F1 은 11항목 중
9개에서 올랐습니다. **국문 성명 0.61 이 최하위**로, 영문 성명(0.77)과 0.16 차이가
나 한글 표기가 구조적 약점입니다.

### 고유식별 4종 하이브리드 (7,689 스팬)

| 구분 | OPF 단독 | 규칙 단독 | **OPF ∪ 규칙** |
| --- | --- | --- | --- |
| 완전 피복 | 7,637 | 7,689 | **7,689** |
| 부분 노출 | 22 | 0 | **0** |
| **완전 미탐** | 30 | 0 | **0** |
| 순수 오탐 스팬 | 5,498 | 0 | 5,498 |

**규칙 레이어 단독으로 4종 전건을 오탐 0 으로 잡습니다.** 하이브리드가 오탐을 늘리지
않고 미탐만 없앱니다.

### 부록 참고치

3자 이상 덮여 TP 지만 안 덮인 나머지로 여전히 식별 가능한 건수 —
argmax 57건 / **Viterbi 30건** (주소 29, 영문 성명 1).

### 프로브셋 300건 (같은 자)

| | argmax | Viterbi |
| --- | --- | --- |
| 재현율 | 0.8067 | 0.8067 |
| F1 | 0.7974 | 0.8359 |
| 금지 전이 | 55 | **0** |

구 실측 197/300 = 0.6567 → **242/300 = 0.8067**. 나빠진 항목 없음.

---

## 실패 모드 축 (3,000건 전수 집계)

`eval/results/showcase.jsonl` 에 축별 대표 5건씩, 좌우 비교용으로 담았습니다.
마스킹은 덮인 문자 1개당 `*` 1개로 **문자 수를 보존**합니다.

| 축 | 모집단 | 정의 |
| --- | --- | --- |
| F1_ORPHAN | 214 | 부분 노출 전체 (symbol 하위 플래그 27) |
| F2_FORM | 238 | 성명 완전미탐 & 앞 15자에 양식 단서 |
| F3_NARR | 444 | 주소 완전미탐 & 앞 15자에 '주소' 없음 |
| F5_MISLABEL | 2,277 | 오프셋 완전일치인데 라벨 불일치 |
| F6_OVERRUN | 2,079 | 경계 과확장 & 초과분에 경칭·직함 |
| F7_INCONSIST | 2,279 | 동일 항목 2개 이상 중 일부만 피복·일부 완전미탐 |
| H1_RULE | 30 | 고유식별 4종 중 OPF 예측과 3자 미달 — 규칙이 구제 |
| H2_OPF_ONLY | 18,222 | 규칙 불가 항목 중 OPF 가 완전 피복 — 규칙으론 못 잡음 |
| V1_DECODE | 56 | argmax 非TP → Viterbi TP |
| D_UNLABELED | 3,768 | 순수오탐 & 예측 라벨 `private_date` |
| S1_STRONG | 1,516 | 정답 스팬 전건 TP 인 문서 |

미탐 축(정답 스팬 28,420)과 과탐 축(순수오탐 5,498)은 **모집단이 달라 합산하지 않습니다.**

### 좌우 비교 HTML 보고서

`eval/results/opf_showcase.html` — 자기완결형 단일 파일(246KB, 11축 55카드).
외부 요청 0건이라 파일만 열면 됩니다.

```bash
python3 eval/build_report_html.py \
  --showcase eval/results/showcase.jsonl --out eval/results/opf_showcase.html
```

원문과 마스킹 결과를 좌우로 놓고 오프셋 기준으로 칠합니다.

| 색 | 위치 | 뜻 |
| --- | --- | --- |
| 파랑 | 원문 | 정답 PII 구간 |
| **빨강** | 원문 | **미탐 — 안 가려진 곳** |
| 회색 | 마스킹 | 덮인 구간 |
| 노랑 | 마스킹 | 과확장 — 정답 밖까지 덮임 |

`H1_RULE`·`H2_OPF_ONLY` 는 규칙 레이어 칸을 더해 3~4단으로, `V1_DECODE`·`F1_ORPHAN`
은 argmax/Viterbi 를 나란히 놓아 3단으로 보여줍니다. 마스킹이 문자 수를 보존하므로
좌우 칸이 같은 위치에서 줄바꿈됩니다.

### 규칙 레이어와 OPF 는 서로를 대체하지 않는다

| | 커버 범위 | OPF 단독 미탐 | 규칙 단독 |
| --- | --- | --- | --- |
| 고유식별 4종 | 7,689 스팬 | 30건 | **전건 탐지, 오탐 0** |
| 나머지 6항목 | 20,731 스팬 | — | **예측 0건** |

규칙은 형식이 법으로 고정된 4종만 잡습니다. 성명·주소·이메일·연락처·계좌·카드는
정규식으로 잡히지 않아 **OPF 가 아니면 가릴 방법이 없습니다**(`H2_OPF_ONLY` 18,222건).
반대로 OPF 가 놓친 4종 30건은 규칙이 전부 구제합니다(`H1_RULE`). 둘을 병용해야
완전 미탐 0 이 됩니다.

### 부분 노출의 원인은 토크나이저가 아니라 태그 시퀀스

argmax 는 `O → I-` 같은 **BIOES 금지 전이를 8,745건** 만듭니다. `group_spans` 가
고아 `I-` 를 스팬 시작으로 관용해 1글자 스팬이 생깁니다.

```
SS-KO-02337  '충청북도 청주시 흥덕구 디지털로 133'
  ...전부 O...  '로'(37,38) I-private_address  ...전부 O...   -> 스팬 '로'
```

제약 Viterbi 는 이 전이를 구조적으로 차단해 **금지 전이 0건**, 1토큰 스팬 1,511 → 3 건.

---

## 로컬 추론 (외부망)

모델 가중치는 **리포 밖** `~/.opf/privacy_filter` 에 둡니다. git 이 추적하지 않습니다.

| 항목 | 값 |
| --- | --- |
| HF repo | `openai/privacy-filter` (public, ungated) |
| 리비전 | `7ffa9a043d54d1be65afb281eddf0ffbe629385b` (2026-04-22) |
| `model.safetensors` sha256 | `06f66b87650b988b04e218285f9fe3df6a4943416b6ffa8171f07bc56cf12a9d` |
| 로딩 | `transformers` 5.15.1 기본 지원 — `trust_remote_code` 불필요 |

```bash
# 단일 문장 — 토큰·오프셋·예측태그 덤프
python3 eval/opf_probe.py --text "My name is Alice Marie Smith"

# 정본 문서 지정 — 정답 스팬 대조까지
python3 eval/opf_probe.py --doc-ids SS-KO-02337 SS-KO-00693

# 속도 실측 / 재현성
python3 eval/opf_probe.py --bench 100
python3 eval/opf_probe.py --repeat 5

# 정본 전량 추론 -> 예측 스팬 JSONL
python3 eval/opf_predict.py --out eval/results/opf_local_preds.jsonl
```

### device 는 CPU 가 기본입니다

| device | 표본 | 건당 | 3,000건 |
| --- | --- | --- | --- |
| **CPU** | 100건 | **0.173초** | **8.6분** |
| MPS | 30건 | 1.062초 | 53.1분 |

MoE 128 experts top-4 의 게더 연산이 잘아 MPS 로 넘기는 비용이 이득을 넘어섭니다.
예측 결과는 두 device 가 20/20 동일해 정확도 손실이 없습니다.

### 검증 (실측)

| 검사 | 결과 |
| --- | --- |
| config 5필드 대조 (모델카드) | `model_type`·`len(id2label)` 33·`n_ctx` 128000·`sliding_window` 128·`hidden_size` 640 전부 일치 |
| 토큰 수 일치 (`logits.shape[0] == len(enc.offsets)`) | **3,000/3,000 통과, 불일치 0건** |
| 재현성 | 동일 입력 5회 태그·스팬 전부 동일 |

`eval/postproc.py` 는 개발망 원본을 그대로 옮긴 것이며 **수정하지 않습니다.**
후처리가 갈리면 개발망 결과와 비교할 수 없습니다.

---

## 재현

```bash
# 정본 재생성 (입력 기본값이 dataset/source/ 를 가리킴)
python3 dataset/build_dataset.py

# 하니스 스키마 변환
python3 dataset/convert_schema.py --gold dataset/ss_pii_testset_ko_v1.json \
  --label opf  --out dataset/eval_opf_labels.jsonl
python3 dataset/convert_schema.py --gold dataset/ss_pii_testset_ko_v1.json \
  --label corp --out dataset/eval_corp_labels.jsonl

# 프로브셋 재생성
python3 eval/scripts/build_probe_set.py

# 로짓 캐시 -> 디코딩 비교 -> 대표 케이스 JSONL (추론은 캐시 생성 때 1회뿐)
python3 eval/cache_logits.py --which gold
python3 eval/cache_logits.py --which probe
python3 eval/decode_compare.py --which gold --out eval/results/showcase.jsonl
python3 eval/decode_compare.py --which probe
```

로짓 캐시는 `~/.opf/cache` (리포 밖)에 둡니다 — 정본 51.4MB + 메타 11.7MB.

시드가 고정되어 있어 재실행 시 바이트 동일한 산출물이 나옵니다.

---

## 실행 환경

| 항목 | 값 |
| --- | --- |
| 장비 | Apple M4 Pro (arm64) |
| 파이썬 | 3.14.3 |
| torch | 2.13.0 (리포 루트 `.venv`, 추적하지 않음) |
| transformers / tokenizers | 5.15.1 / 0.22.2 (`.venv` 안에만 설치) |
| device | **cpu** — cuda 불가, mps 는 가능하나 6배 느림 |

로컬 추론은 CPU 가 기본입니다 (위 **로컬 추론** 절 참고).
`opf eval` 하니스를 쓸 때는 기본 device 가 cuda라 `--device mps` 를 명시해야 합니다.
집계 스크립트 9종은 표준 라이브러리만 쓰므로 torch 없이 동작합니다.

```bash
source .venv/bin/activate
```

---

## 다음 실험

1~3 은 완료했습니다(라벨 무시 마스킹, Viterbi 디코딩, 3,000건 전량 집계).
남은 것은 아래입니다.

1. **Viterbi 운영점 조정** — `viterbi_calibration.json` 의 6개 transition-bias 가
   현재 전부 `0.0`(default 운영점)이라 순수 구조 제약만 걸려 있습니다. 모델카드
   §2.3.2 는 이 값으로 재현율/정밀도 운영점을 연속 조정할 수 있다고 합니다.
   파일에 default 하나뿐이라 값을 지어내지 않았습니다.
2. **국문 성명 재현율 0.6149** — 11항목 중 최하위이고 영문 성명(0.7675)과 0.16
   차이입니다. `F2_FORM`(238건) 축이 양식 단서 뒤 미탐을 모아 둔 재료입니다.
3. **주소 경계** — `F6_OVERRUN`(2,079건)과 `F3_NARR`(444건)이 같은 항목에서
   과확장과 완전미탐이 동시에 나는 것을 보여줍니다.
4. **고유식별 4종은 규칙 레이어로 종결** — OPF ∪ 규칙이 완전 미탐 0 을 달성했으므로
   이 4종은 추가 모델 작업이 불필요합니다.

개발망 하니스로 돌릴 때는 `eval/RUNBOOK.txt` 에서 `$MODEL` 만 실제 경로로 바꾸고
`[0]` 부터 순서대로 실행합니다.
