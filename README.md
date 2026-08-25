# 2026S AI Solution OJT — OPF 한국어 PII 평가

OpenAI Privacy Filter(OPF)를 사내 도입할 수 있는지 판단하기 위해,
한국어 PII 테스트셋을 만들고 그것으로 모델을 평가하는 과제입니다.

- **`dataset/`** — 평가할 데이터를 만든다 (1주차)
- **`eval/`** — 그 데이터로 모델을 평가한다 (2주차)

모델 체크포인트를 외부망 로컬에 반입해 **직접 추론이 가능합니다.** 가중치는
`~/.opf/privacy_filter` (리포 밖, 추적하지 않음)에 있고, `eval/opf_local.py` 가
추론 어댑터입니다. 개발망 vLLM `/pooling` 경로는 스위치로만 남겨 두었습니다.

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
opf_local.py   로컬 추론 어댑터 — str -> (spans, masked_text)
opf_probe.py   토큰·오프셋·예측태그 전수 덤프 (원인 규명용)
opf_predict.py 정본 전량 추론 -> 예측 스팬 JSONL
postproc.py    group_spans / mask_text (개발망 원본 전사, 수정 금지)
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

## OPF 실측 관찰 (문서 5건 예비)

`SS-KO-02337 / 00693 / 00005 / 00003 / 00001` 을 토큰 단위로 덤프해 얻은 관찰입니다.
**표본 5건이라 경향이며, 3,000건 집계로 확정해야 합니다.**

### 잘하는 것

이메일·연락처·주민/외국인등록번호·여권·운전면허·카드·계좌 등 **형식이 뚜렷한 항목**은
문맥이 정상이면 5건 전부 완전탐지했습니다. 영문 성명도 3/3 탐지했습니다.

### 못하는 것

**1. 국문 성명 미탐** — 5개 문서 국문 성명 9건 중 **6건 미탐**. 영문 성명은 3/3 탐지라
한글 표기 자체가 약점입니다. 같은 문서 안에서 `- 이름:` 뒤는 잡고 `성명(` 뒤는 놓쳐,
한국어 라벨 단서 커버리지가 고르지 않습니다.

**2. 부분노출의 원인은 토크나이저가 아니라 태그 시퀀스** — 주소·카드번호가 1글자만
마스킹되는 사례는 토큰 경계 문제가 아닙니다. 모델이 해당 구간을 전부 `O` 로 찍고
고아 `I-` 태그 하나만 남기면, `group_spans` 가 그 토큰만으로 스팬을 열기 때문입니다.

```
SS-KO-02337  '충청북도 청주시 흥덕구 디지털로 133'
  ...전부 O...  '로'(37,38) I-private_address  ...전부 O...   -> 스팬 '로'

SS-KO-00693  '$5425-1253-2994-9092'   ($ 때문에 금액으로 읽힘)
  ...전부 O...  '-'(164,165) I-account_number  ...전부 O...   -> 스팬 '-'
```

**3. 과탐지유도 함정에 그대로 걸림** — 정본이 정답 스팬 없이 심어둔 종목코드·주문번호를
`account_number` 로 잡았습니다 (`종목코드 035420`, `주문번호 20260575-549159` 등).

**4. 한국어 일반명사를 인명으로** — `예산 현황`·`민간`·`후원` 을 `private_person` 으로.

**5. 라벨 불안정** — 운전면허번호 `12-50-894573-19` 하나가 4조각으로 갈리며
`account_number` / `private_phone` 이 번갈아 붙습니다.

**6. 경계 과확장** — 조사·직함까지 삼킵니다 (`MYEONGHUI KIM님`, ` PB 김상우 수석`).
BPE 토큰이 앞 공백을 포함해 마스킹이 공백을 먹는 것도 같은 계열입니다
(`My name is[private_person]`).

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
```

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

로컬 추론이 되므로 아래를 외부망에서 직접 돌릴 수 있습니다.

1. **3,000건 전량 집계** — `opf_predict.py` 로 예측 JSONL 을 만들고
   `run_all.py --pred` 로 기존 집계 스크립트에 태웁니다 (추론 약 9분).
2. **라벨 무시 마스킹** — 라벨 종류가 흔들려도 구간만 맞으면 마스킹은 성립합니다.
   `account_number` / `private_phone` 혼동이 실제 미탐인지 표기 문제인지 가릅니다.
3. **Viterbi 디코딩** — 현재는 argmax 단독이라 고아 `I-` 태그가 1글자 스팬을
   만듭니다. `viterbi_calibration.json` 이 체크포인트에 함께 있습니다.
4. **주소 경계 확장** — 두 주소를 하나로 병합하는 과확장과, 반대로 전부 놓치는
   미탐이 같은 문서에서 나옵니다.

개발망 하니스로 돌릴 때는 `eval/RUNBOOK.txt` 에서 `$MODEL` 만 실제 경로로 바꾸고
`[0]` 부터 순서대로 실행합니다.
