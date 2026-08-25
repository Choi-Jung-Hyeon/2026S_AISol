# 2026S AI Solution OJT — OPF 한국어 PII 평가

OpenAI Privacy Filter(OPF)를 사내 도입할 수 있는지 판단하기 위해,
한국어 PII 테스트셋을 만들고 그것으로 모델을 평가하는 과제입니다.

- **`dataset/`** — 평가할 데이터를 만든다 (1주차)
- **`eval/`** — 그 데이터로 모델을 평가한다 (2주차)

모델 체크포인트는 아직 반입 전이라, 현재 리포에는 **모델 없이 준비 가능한 것**까지
들어 있습니다. 반입 후에는 `eval/RUNBOOK.txt`만 따라 실행하면 됩니다.

---

## 디렉토리

```
dataset/    정본 테스트셋, 생성기, 가명값 풀, 원천 입력(source/)
eval/       평가 파이프라인 — 진입점·집계 스크립트·운영점·실행 절차
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
data/       정본·eval 라벨·스모크 미니셋·부분 발화 프로브셋
scripts/    집계 스크립트 9종
configs/    Viterbi 운영점 3종
results/    실행 결과 (예측 JSONL 등은 재생성 가능하므로 추적하지 않음)
RUNBOOK.txt 실행 순서 [0]~[7]
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
| device | **mps** — cuda 불가 |

`opf eval`의 기본 device는 cuda라 매 명령에 `--device mps`를 명시해야 합니다.
집계 스크립트 9종은 표준 라이브러리만 쓰므로 torch 없이 동작합니다.

```bash
source .venv/bin/activate
```

---

## 모델 반입 후

`eval/RUNBOOK.txt`에서 `$MODEL`만 실제 경로로 바꾸고 `[0]`부터 순서대로
실행합니다. `[1]` 스모크 50건 결과로 각 단계의 `# est: ___`를 채운 뒤
`[2]` 기준선 3,000건으로 넘어가는 흐름입니다.
