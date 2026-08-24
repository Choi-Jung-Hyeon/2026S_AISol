#!/usr/bin/env python3
"""OPF 한국어 PII 평가 — 단일 진입점.

개발망은 망 분리로 결과 반출이 어렵다. 자리에서 화면을 그대로 보여주는 것이
보고 수단이므로, 명령 하나로 6단계를 순서대로 돌리고 출력을 한 흐름으로 모은다.

  python3 run_all.py --model <모델경로>

각 단계는 scripts/ 의 스크립트를 subprocess 로 부른다. 지표 계산 로직을
여기에 복사하지 않는다 — 수치의 출처는 언제나 해당 스크립트 하나뿐이다.
콘솔에 나가는 모든 줄은 results/REPORT_<YYYYMMDD_HHMM>.txt 에 동시에 적힌다.

run_opf.py 는 개발망에만 있다. 이 리포에는 없으므로 [1/6] 에서 부재를 알리고
멈춘다. 예측 파일이 이미 있으면 --skip-opf 또는 --pred 로 [1/6] 을 건너뛴다.

표준 라이브러리만 사용한다.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
GOLD_FULL = os.path.join(DATA, "ss_pii_testset_ko_v1.json")
EVAL_DATA = os.path.join(DATA, "eval_corp_labels.jsonl")
SMOKE_DATA = os.path.join(DATA, "smoke_corp_50.jsonl")
RUN_OPF = os.path.join(HERE, "run_opf.py")
# --skip-opf 로 재사용할 기본 예측 (RUNBOOK [2] 의 untyped 산출물)
DEFAULT_PRED = os.path.join(RESULTS, "baseline_untyped_preds.jsonl")
SMOKE_N = 50
WIDTH = 80

STEPS = [
    ("1/6", "OPF 추론", "run_opf.py"),
    ("2/6", "성능지표 (표준 정의)", "standalone_metrics.py"),
    ("3/6", "문서 단위 미탐", "doc_level_miss.py"),
    ("4/6", "규칙 레이어", "rule_layer.py"),
    ("5/6", "하이브리드 비교", "hybrid_merge.py"),
    ("6/6", "미탐·과탐 예시", "dump_examples.py"),
]

# 단계별 복구 안내 1줄 (A-3)
RECOVERY = {
    "1/6": "run_opf.py 를 직접 실행해 예측 JSONL 을 만든 뒤 "
           "--skip-opf --pred <예측경로> 로 재실행하십시오.",
    "2/6": "예측 JSONL 의 text 가 정본과 한 글자도 다르면 조인이 실패합니다 — "
           "같은 data/ 파일로 추론했는지 확인하십시오.",
    "3/6": "예측 JSONL 경로와 형식(하니스 predictions 키)을 확인하십시오.",
    "4/6": "results/ 디렉토리 쓰기 권한과 정본 JSON 경로를 확인하십시오.",
    "5/6": "OPF 예측과 규칙 예측이 같은 문서 집합인지 확인하십시오.",
    "6/6": "정본·예측·규칙 세 파일이 모두 같은 문서 집합인지 확인하십시오.",
}


# ── 폭 보정 출력 ─────────────────────────────────────────────
def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s, w):
    s = str(s)
    return s + " " * max(0, w - dwidth(s))


class Tee(object):
    """콘솔과 보고서 파일에 같은 바이트를 동시에 쓴다."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.f.write(s)
        self.f.flush()

    def close(self):
        self.f.close()


TEE = None


def out(line=""):
    TEE.write(line + "\n")


def rule(ch="="):
    out(ch * WIDTH)


def die(msg, code=1):
    out("")
    rule()
    for ln in msg.split("\n"):
        out(ln)
    rule()
    if TEE is not None:
        TEE.close()
    raise SystemExit(code)


def fracpct(n, d):
    pct = "n/a" if not d else "%.2f%%" % (100.0 * n / d)
    return "{:,} / {:,} ({})".format(n, d, pct)


# ── 단계 실행 ────────────────────────────────────────────────
def run_step(tag, title, cmd):
    """하위 스크립트를 실행하고 출력을 그대로 흘려보낸다.

    돌려주는 값: (소요 초, 출력 줄 목록). 실패하면 그 자리에서 중단한다.
    """
    out("")
    rule()
    out("[%s] %s" % (tag, title))
    rule()
    out("명령: %s" % " ".join(cmd))
    out("")
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    lines = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=HERE, env=env,
                                universal_newlines=True, bufsize=1)
    except OSError as e:
        die("*** [%s] %s 실행 실패 — %s\n"
            "실패한 명령:\n  %s\n복구: %s" % (tag, title, e, " ".join(cmd),
                                          RECOVERY[tag]), 1)
    for line in proc.stdout:
        line = line.rstrip("\n")
        out(line)
        lines.append(line)
    rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        die("*** [%s] %s 실패 (exit %d, %.1f초)\n"
            "실패한 명령:\n  %s\n"
            "복구: %s\n"
            "이후 단계는 실행하지 않았습니다."
            % (tag, title, rc, dt, " ".join(cmd), RECOVERY[tag]), rc)
    out("")
    out("[%s] 완료 — %.1f초" % (tag, dt))
    return dt, lines


def skip_step(tag, title, reason):
    out("")
    rule()
    out("[%s] %s" % (tag, title))
    rule()
    out("생략 — %s" % reason)
    return 0.0, []


# ── 스모크용 정본 부분집합 ───────────────────────────────────
def make_smoke_gold(dst):
    """정본 앞 50건만 담은 부분집합을 만든다.

    스모크 데이터(smoke_corp_50.jsonl)는 정본 앞 50건과 같은 문서이므로,
    정본 3,000건을 그대로 쓰면 2,950건이 조인 실패해 [2/6] 이 멈춘다.
    정본 파일 자체는 건드리지 않고 results/ 에 파생본만 만든다.
    """
    with open(GOLD_FULL, encoding="utf-8") as f:
        data = json.load(f)
    docs = data.get("documents") if isinstance(data, dict) else data
    sub = docs[:SMOKE_N]
    if isinstance(data, dict):
        data = dict(data)
        data["documents"] = sub
    else:
        data = sub
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return len(sub)


# ── 요약 블록 ────────────────────────────────────────────────
def grab(lines, pattern, group=1):
    """출력 줄에서 정규식으로 값 하나를 집는다. 못 찾으면 None."""
    rx = re.compile(pattern)
    for ln in lines:
        m = rx.search(ln)
        if m:
            return m.group(group)
    return None


def to_int(s):
    return None if s is None else int(s.replace(",", ""))


def summary(metrics_path, doc_lines, hybrid_lines, dump_lines, timings, total):
    out("")
    rule()
    out("요약 — 이 블록만으로 보고할 수 있어야 한다")
    rule()

    try:
        with open(metrics_path, encoding="utf-8") as f:
            M = json.load(f)
    except (OSError, ValueError) as e:
        out("지표 JSON 을 읽지 못했습니다: %s (%s)" % (metrics_path, e))
        M = None

    def line(label, value):
        out("%s : %s" % (pad(label, 22), value))

    line("측정 단위", "스팬 (문자 오프셋 [start, end))")
    if M:
        o = M["overall"]
        st = M["standard_criteria"]["strict"]
        line("TP / FN / FP", "{:,} / {:,} / {:,}".format(o["tp"], o["fn"], o["fp"]))
        out("%s   (partial 기준 — 재현율·정밀도가 공유하는 단일 분자)" % pad("", 22))
        line("재현율", fracpct(o["recall"]["num"], o["recall"]["den"]))
        line("정밀도", fracpct(o["precision"]["num"], o["precision"]["den"]))
        line("F2 (0~1 비율)", "%.4f" % o["f2"])
        line("Strict 재현율 (하한)", fracpct(st["recall"]["num"], st["recall"]["den"]))
        line("Strict 정밀도 (하한)", fracpct(st["precision"]["num"], st["precision"]["den"]))
        line("과잉 마스킹", "{:,}자".format(o["over_masked_chars"]))
        dl = M["doc_level"]
        line("문서 단위 미탐", fracpct(dl["with_miss"], dl["total"]))
    else:
        for lb in ("TP / FN / FP", "재현율", "정밀도", "F2 (0~1 비율)",
                   "Strict 재현율 (하한)", "Strict 정밀도 (하한)",
                   "과잉 마스킹", "문서 단위 미탐"):
            line(lb, "(지표 JSON 없음)")

    before = to_int(grab(hybrid_lines, r"변경 전.*잔여 미탐\s*:\s*([\d,]+)건"))
    after = to_int(grab(hybrid_lines, r"변경 후.*잔여 미탐\s*:\s*([\d,]+)건"))
    if before is None or after is None:
        line("고유식별 4종 잔여 미탐", "(하이브리드 출력에서 확인 불가)")
    else:
        line("고유식별 4종 잔여 미탐",
             "병합 전 {:,}건 / 병합 후 {:,}건".format(before, after))

    shown = {}
    rx = re.compile(r"^\[([A-F])\]\s+\S.*?:\s*([\d,]+)건 중 ([\d,]+)건 표시")
    for ln in dump_lines:
        m = rx.search(ln)
        if m:
            shown[m.group(1)] = m.group(3).replace(",", "")
    line("예시 출력 건수 (A~F)",
         " / ".join("%s %s" % (b, shown.get(b, "-")) for b in "ABCDEF"))
    rule()

    out("")
    out("단계별 소요 (초)")
    for (tag, title, _), dt in zip(STEPS, timings):
        out("  [%s] %s %s" % (tag, pad(title, 22), "생략" if dt == 0.0
                              else "%8.1f" % dt))
    out("  %s %s" % (pad("전체", 27), "%8.1f" % total))


def main():
    global TEE
    ap = argparse.ArgumentParser(
        prog="run_all.py",
        description="OPF 한국어 PII 평가 6단계를 한 번에 실행한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="사용법\n"
               "  python3 run_all.py --model <모델경로>\n"
               "\n"
               "예측 파일이 이미 있으면 추론을 건너뛴다\n"
               "  python3 run_all.py --skip-opf\n"
               "  python3 run_all.py --pred results/baseline_untyped_preds.jsonl\n"
               "\n"
               "모든 출력은 results/REPORT_<YYYYMMDD_HHMM>.txt 에도 적힌다.")
    ap.add_argument("--model", default=None,
                    help="OPF 모델 경로 (run_opf.py 에 그대로 넘긴다)")
    ap.add_argument("--smoke", action="store_true",
                    help="스모크 %d건만 실행 (기본 꺼짐)" % SMOKE_N)
    ap.add_argument("--skip-opf", action="store_true",
                    help="OPF 재실행을 생략하고 기존 예측을 재사용한다")
    ap.add_argument("--pred", default=None,
                    help="예측 JSONL 을 직접 지정한다 (지정 시 추론 생략)")
    args = ap.parse_args()

    if not os.path.isdir(RESULTS):
        sys.stderr.write("[run_all] results 디렉토리가 없습니다: %s\n" % RESULTS)
        raise SystemExit(1)
    stamp = time.strftime("%Y%m%d_%H%M")
    report = os.path.join(RESULTS, "REPORT_%s.txt" % stamp)
    # 같은 분에 두 번 돌리면 앞 보고서를 덮어쓴다. 반출이 막힌 자리에서
    # 직전 실행 기록을 잃지 않도록 충돌할 때만 꼬리번호를 붙인다.
    n = 2
    while os.path.exists(report):
        report = os.path.join(RESULTS, "REPORT_%s_%d.txt" % (stamp, n))
        n += 1
    TEE = Tee(report)

    t_all = time.time()
    rule()
    out("OPF 한국어 PII 평가 — 전체 실행")
    rule()
    out("실행 시각   : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    out("실행 위치   : %s" % HERE)
    out("보고서 파일 : %s" % report)
    out("범위        : %s" % ("스모크 %d건" % SMOKE_N if args.smoke
                              else "전체 3,000건"))
    out("모드        : %s" % ("예측 직접 지정 (--pred)" if args.pred
                              else ("기존 예측 재사용 (--skip-opf)" if args.skip_opf
                                    else "OPF 추론부터 실행")))
    out("")
    out("화면에 나온 모든 줄은 위 보고서 파일에 그대로 적힙니다.")
    out("망 분리로 파일 반출이 어려우면 이 화면을 그대로 보여주면 됩니다.")

    # 스모크면 정본 부분집합을 만들어 이후 단계의 조인을 맞춘다
    gold = GOLD_FULL
    if args.smoke:
        gold = os.path.join(RESULTS, "_smoke_gold_%d.json" % SMOKE_N)
        try:
            n = make_smoke_gold(gold)
        except (OSError, ValueError) as e:
            die("스모크용 정본 부분집합을 만들지 못했습니다: %s" % e, 1)
        out("")
        out("스모크 정본 부분집합 생성 : %s (%d건)" % (gold, n))
        out("  정본 원본은 건드리지 않습니다. 조인을 맞추기 위한 파생본입니다.")

    timings = []

    # ── [1/6] OPF 추론 ──────────────────────────────────────
    tag, title, _ = STEPS[0]
    if args.pred:
        pred = args.pred if os.path.isabs(args.pred) else os.path.join(HERE, args.pred)
        if not os.path.isfile(pred):
            out("")
            rule()
            out("[%s] %s" % (tag, title))
            rule()
            die("*** [%s] %s 실패 — 지정한 예측 파일이 없습니다.\n"
                "  --pred %s\n"
                "복구: 경로를 확인하거나, 예측이 없으면 --model 로 추론부터 "
                "실행하십시오.\n"
                "이후 단계는 실행하지 않았습니다." % (tag, title, args.pred), 1)
        dt, _ = skip_step(tag, title, "--pred 로 지정한 예측을 씁니다: %s" % pred)
    elif args.skip_opf:
        pred = DEFAULT_PRED
        if not os.path.isfile(pred):
            out("")
            rule()
            out("[%s] %s" % (tag, title))
            rule()
            die("*** [%s] %s 실패 — 재사용할 예측 파일이 없습니다.\n"
                "  찾은 경로: %s\n"
                "복구: --pred <예측 JSONL 경로> 로 직접 지정하십시오.\n"
                "이후 단계는 실행하지 않았습니다." % (tag, title, pred), 1)
        dt, _ = skip_step(tag, title, "기존 예측을 재사용합니다: %s" % pred)
    else:
        if not args.model:
            out("")
            rule()
            out("[%s] %s" % (tag, title))
            rule()
            die("*** [%s] %s 실패 — --model 이 없습니다.\n"
                "  사용법: python3 run_all.py --model <모델경로>\n"
                "복구: 모델 경로를 주거나, 예측이 이미 있으면 "
                "--skip-opf 또는 --pred <경로> 를 쓰십시오.\n"
                "이후 단계는 실행하지 않았습니다." % (tag, title), 1)
        if not os.path.exists(args.model):
            out("")
            rule()
            out("[%s] %s" % (tag, title))
            rule()
            die("*** [%s] %s 실패 — 모델 경로가 없습니다: %s\n"
                "복구: ls 로 경로를 확인한 뒤 --model 에 다시 주십시오.\n"
                "이후 단계는 실행하지 않았습니다." % (tag, title, args.model), 1)
        if not os.path.isfile(RUN_OPF):
            out("")
            rule()
            out("[%s] %s" % (tag, title))
            rule()
            die("*** [%s] %s 실패 — run_opf.py 가 없습니다.\n"
                "  찾은 경로: %s\n"
                "  run_opf.py 는 개발망에만 있고 이 리포에는 포함되지 않습니다.\n"
                "복구: 예측 JSONL 이 이미 있으면 --skip-opf, 다른 위치에 있으면 "
                "--pred <경로> 로 실행하십시오.\n"
                "이후 단계는 실행하지 않았습니다." % (tag, title, RUN_OPF), 1)
        pred = os.path.join(RESULTS, "opf_predictions_%s.jsonl" % stamp)
        data_in = SMOKE_DATA if args.smoke else EVAL_DATA
        dt, _ = run_step(tag, title,
                         [sys.executable, RUN_OPF, "--model", args.model,
                          "--data", data_in, "--out", pred])
        if not os.path.isfile(pred):
            die("*** [%s] %s 실패 — 예측 파일이 만들어지지 않았습니다: %s\n"
                "복구: run_opf.py 의 출력 인자 이름이 --out 이 아닐 수 있습니다. "
                "직접 실행한 뒤 --pred <경로> 로 재실행하십시오.\n"
                "이후 단계는 실행하지 않았습니다." % (tag, title, pred), 1)
    timings.append(dt)

    # ── [2/6] 성능지표 ──────────────────────────────────────
    metrics_json = os.path.join(RESULTS, "_metrics_%s.json" % stamp)
    dt, _ = run_step(STEPS[1][0], STEPS[1][1],
                     [sys.executable, os.path.join(SCRIPTS, "standalone_metrics.py"),
                      "--gold", gold, "--pred", pred,
                      "--out-json", metrics_json, "--partial-sample", "10"])
    timings.append(dt)

    # ── [3/6] 문서 단위 미탐 ────────────────────────────────
    dt, doc_lines = run_step(STEPS[2][0], STEPS[2][1],
                             [sys.executable, os.path.join(SCRIPTS, "doc_level_miss.py"),
                              "--gold", gold, "--predictions", pred,
                              "--sample", "20"])
    timings.append(dt)

    # ── [4/6] 규칙 레이어 ───────────────────────────────────
    # 기존 results/rule_predictions.jsonl 을 덮지 않도록 타임스탬프 경로로 쓴다
    rule_out = os.path.join(RESULTS, "rule_predictions_%s.jsonl" % stamp)
    dt, _ = run_step(STEPS[3][0], STEPS[3][1],
                     [sys.executable, os.path.join(SCRIPTS, "rule_layer.py"),
                      "--gold", gold, "--out", rule_out])
    timings.append(dt)

    # ── [5/6] 하이브리드 비교 ───────────────────────────────
    dt, hybrid_lines = run_step(STEPS[4][0], STEPS[4][1],
                                [sys.executable, os.path.join(SCRIPTS, "hybrid_merge.py"),
                                 "--gold", gold, "--opf", pred,
                                 "--rule", rule_out, "--focus", "고유식별정보"])
    timings.append(dt)

    # ── [6/6] 미탐·과탐 예시 ────────────────────────────────
    dt, dump_lines = run_step(STEPS[5][0], STEPS[5][1],
                              [sys.executable, os.path.join(SCRIPTS, "dump_examples.py"),
                               "--gold", gold, "--pred", pred,
                               "--rule", rule_out, "--n", "5"])
    timings.append(dt)

    total = time.time() - t_all
    summary(metrics_json, doc_lines, hybrid_lines, dump_lines, timings, total)

    out("")
    out("보고서 파일 : %s" % report)
    out("이 화면과 파일 내용은 같습니다.")
    TEE.close()


if __name__ == "__main__":
    main()
