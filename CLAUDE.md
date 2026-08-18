# CLAUDE.md

OPF 한국어 PII 평가 OJT 리포. 전체 개요는 `README.md` 를 먼저 읽으십시오.
이 파일은 **모르면 사고가 나는 규칙**만 담습니다.

## 브랜치

작업 브랜치는 `eval-schema-conversion` 입니다. **`main` 에서 직접 작업하지 마십시오.**
이 리포는 한 브랜치를 계속 재사용하며 PR 이 순차로 머지됩니다.
따라서 푸시 전에 직전 PR 이 이미 머지됐는지 확인하고, 머지됐으면 새 PR 을 만듭니다.
머지 방식은 머지 커밋(`gh pr merge --merge`)으로 기존 이력에 맞춥니다.

## 정본 데이터셋은 손으로 고치지 않습니다

`과제1-2/ss_pii_testset_ko_v1.json` 이 정본입니다. 값이나 오프셋을 직접 편집하지 말고
**항상 `build_dataset.py` 를 고쳐 재생성**하십시오. 파생 산출물은 아래 순서로 함께
갱신해야 하며, 하나라도 빠지면 정본과 어긋납니다.

```
build_dataset.py
  -> ss_pii_testset_ko_v1.json / _opf.jsonl / pseudonym_pool.json
convert_schema.py
  -> eval_opf_labels.jsonl / eval_corp_labels.jsonl
과제2-1/data 사본 복사 (과제1-2 와 SHA256 일치해야 함)
head -n 50  -> smoke_opf_50.jsonl / smoke_corp_50.jsonl
build_probe_set.py -> probe_partial_utterance{,_corp}.jsonl
rule_layer.py      -> results/rule_predictions{,_probe}.jsonl
```

입력 경로 주의 — `openpii_ko_sample.jsonl` 이 `archive/` 로 이동해 기본값으로는
실행되지 않습니다. **`SS_SRC` 를 반드시 지정**하십시오.

```bash
SS_SRC="$PWD/archive/openpii_ko_sample.jsonl" python3 과제1-2/build_dataset.py
```

## 생성기를 고칠 때 — 난수 스트림을 깨지 마십시오

`build_dataset.py`(seed 20260814), `build_probe_set.py`(seed 20260818) 는
재실행 시 **바이트 동일한 산출물**이 나와야 합니다.

- **`random` 호출을 추가·제거·순서 변경 하지 마십시오.** 재추첨 루프를 넣으면
  스트림이 어긋나 성명·주소 등 무관한 값까지 전부 달라집니다.
  값을 바꿔야 하면 **이미 뽑은 난수를 결정적으로 사상**하십시오.
  (예: `rrn_invalid()` 에서 오프셋 2를 3으로 옮김 — 호출 횟수 불변)
- **값의 자릿수를 바꾸지 마십시오.** 길이가 변하면 문서 내 후속 스팬 오프셋이
  전부 밀립니다.
- 수정 **전에** 현재 스크립트가 커밋된 산출물을 바이트 동일하게 재현하는지
  먼저 확인하십시오. 이 기준선이 없으면 이후 차이를 수정 탓으로 돌릴 근거가 없습니다.

## 평가 지표 규칙

미탐(FN)이 가장 치명적입니다. **Recall > F2 > F1 > Precision**.
Accuracy 는 배경 토큰 비중 때문에 과대평가되어 주지표에서 제외합니다.

- **스팬 매칭은 비대칭입니다.** 재현율 TP 는 `정답 ⊆ 예측`, 정밀도 TP 는
  `예측 ⊆ 정답` 이며 **두 TP 개수가 다릅니다.** 두 축을 섞지 마십시오.
- **FN 과 FP 를 합산한 값을 만들지 마십시오.** 서로 다른 모집단입니다.
- **조인은 text 완전 일치로만** 합니다. `opf eval --predictions-out` 의
  `example_id` 는 sha256 자동 생성값이라 정본 id 와 다릅니다.
  조인 실패 건수를 출력하고, 0 이 아니면 집계를 중단하십시오.
- **백분율을 쓰지 마십시오.** recall/precision 은 분수(`4/4900`),
  f1/f2 는 조화평균이라 분수로 환원되지 않으므로 0~1 비율로 씁니다.
- `ground_truth_label_recall` 에서 되살린 건수는 반올림 근사치입니다.
  확정 미탐 건수는 `doc_level_miss.py` 가 예측 스팬을 직접 세어 산출합니다.

## 스크립트 제약

`과제2-1/scripts/` 6종은 **표준 라이브러리만** 씁니다. torch/numpy 등 외부
의존성을 추가하지 마십시오. 입력 파일 부재 시 스택트레이스 대신 명확한 메시지와
exit code 로 종료해야 합니다.

## 실행 환경

`.venv` 에 torch 2.13.0 이 설치되어 있습니다(추적하지 않음). 이 장비는 **Apple
Silicon 이라 cuda 를 쓸 수 없고 device 는 `mps`** 입니다. `opf eval` 기본값이
cuda 라 매 명령에 `--device mps` 를 명시해야 합니다.

공용 장비이며 `/opt/homebrew` 가 타 사용자 소유입니다. **sudo·brew·전역 설치를
쓰지 마십시오.** 패키지는 `.venv` 안에만 설치합니다.

## 커밋에서 제외하는 것

- `과제2-1/results/*.jsonl` — 커밋된 스크립트로 재생성되는 결정적 산출물
- `.venv/`
- 내가 만들지 않은 변경 — 작업 트리에 타인의 수정이 섞여 있으면 스테이징에서
  빼고 보고하십시오. `git add <dir>` 로 뭉뚱그리지 말고 경로를 지정하십시오.

## 산출물 주장은 실측으로 뒷받침합니다

"통과했습니다" 같은 서술 대신 **실제 수치**를 출력하십시오. 문서에 수치를 적을
때는 데이터에서 다시 세어 대조합니다. 정본 SHA256 이 바뀌면 이를 참조하는 문서도
함께 갱신해야 합니다.
