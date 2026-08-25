"""제약 Viterbi 디코더 — BIOES 허용 전이만 통과시키는 linear-chain CRF 디코딩.

postproc.py 는 건드리지 않는다. 이 모듈이 별도 경로다.

근거는 공식 모델카드 §2.3.1 과 리포에 동봉된 공식 구현
reference/opf/opf/_core/decoding.py 의 ViterbiCRFDecoder 다.
start / transition / end 는 캘리브레이션 파일에 없다 — BIOES 라벨 공간에서
런타임에 유도하는 것이 정상이다. 파일에는 6개 transition-bias 만 들어 있고,
그 6개만 실제로 파일에서 읽어 쓴다. 없는 파라미터를 지어내지 않는다.

argmax 단독은 O -> I- 같은 금지 전이를 허용해 1글자 스팬을 만든다.
postproc.group_spans 는 그 고아 I- 를 스팬 시작으로 관용한다 —
이 모듈의 spans_from_bioes 는 BIOES 를 그대로 해석해 그러지 않는다.
"""

from __future__ import annotations

import json
import os

import numpy as np

NEG_INF = -1e9

BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


def load_biases(ckpt):
    """viterbi_calibration.json 의 default 운영점에서 6개 bias 를 읽는다."""
    path = os.path.join(ckpt, "viterbi_calibration.json")
    if not os.path.isfile(path):
        raise SystemExit("viterbi_calibration.json 이 없습니다: %s" % path)
    with open(path, encoding="utf-8") as f:
        art = json.load(f)
    try:
        raw = art["operating_points"]["default"]["biases"]
    except (KeyError, TypeError):
        raise SystemExit("operating_points.default.biases 를 찾을 수 없습니다")
    missing = [k for k in BIAS_KEYS if k not in raw]
    if missing:
        raise SystemExit("bias 키 누락 %s — 임의로 지어내지 않고 중단합니다" % missing)
    extra = [k for k in raw if k not in BIAS_KEYS]
    if extra:
        raise SystemExit("알 수 없는 bias 키 %s" % extra)
    return {k: float(raw[k]) for k in BIAS_KEYS}, path


def parse_labels(id2label):
    """id -> (접두사, 카테고리). O 는 (None, None)."""
    n = len(id2label)
    tag = [None] * n
    cat = [None] * n
    bg = None
    for i in range(n):
        t = id2label[str(i)]
        if t == "O":
            bg = i
            continue
        p, c = t.split("-", 1)
        tag[i] = p
        cat[i] = c
    if bg is None:
        raise SystemExit("id2label 에 배경 라벨 O 가 없습니다")
    return tag, cat, bg


def is_valid_transition(prev_tag, prev_cat, next_tag, next_cat):
    """공식 _is_valid_transition 과 같은 규칙.

        start / O / E- / S-  ->  O, B-*, S-*
        B-X / I-X            ->  I-X, E-X  (같은 카테고리만)
    """
    next_bg = next_tag is None
    if prev_tag is None:                       # 이전이 O(배경)
        return next_bg or next_tag in ("B", "S")
    if prev_tag in ("E", "S"):                 # 스팬이 닫혀 있다
        return next_bg or next_tag in ("B", "S")
    if prev_tag in ("B", "I"):                 # 스팬이 열려 있다
        return (not next_bg) and prev_cat == next_cat and next_tag in ("I", "E")
    return False


def transition_bias(prev_tag, prev_cat, next_tag, next_cat, b):
    """공식 _transition_bias 와 같은 매핑."""
    next_bg = next_tag is None
    if prev_tag is None:
        if next_bg:
            return b["transition_bias_background_stay"]
        if next_tag in ("B", "S"):
            return b["transition_bias_background_to_start"]
        return 0.0
    if prev_tag in ("B", "I"):
        if next_tag == "I" and prev_cat == next_cat:
            return b["transition_bias_inside_to_continue"]
        if next_tag == "E" and prev_cat == next_cat:
            return b["transition_bias_inside_to_end"]
        return 0.0
    if prev_tag in ("E", "S"):
        if next_bg:
            return b["transition_bias_end_to_background"]
        if next_tag in ("B", "S"):
            return b["transition_bias_end_to_start"]
        return 0.0
    return 0.0


class ConstrainedViterbi:
    """BIOES 제약 linear-chain Viterbi."""

    def __init__(self, id2label, biases):
        self.id2label = id2label
        self.n = len(id2label)
        self.tag, self.cat, self.bg = parse_labels(id2label)
        self.biases = biases

        n = self.n
        self.start = np.full(n, NEG_INF, dtype=np.float64)
        self.end = np.full(n, NEG_INF, dtype=np.float64)
        self.trans = np.full((n, n), NEG_INF, dtype=np.float64)

        for i in range(n):
            t = self.tag[i]
            # 시퀀스는 O / B- / S- 로만 시작할 수 있다
            if t in ("B", "S") or i == self.bg:
                self.start[i] = 0.0
            # 시퀀스는 O / E- / S- 로만 끝날 수 있다 (열린 스팬을 남기지 않는다)
            if t in ("E", "S") or i == self.bg:
                self.end[i] = 0.0
            for j in range(n):
                if is_valid_transition(t, self.cat[i], self.tag[j], self.cat[j]):
                    self.trans[i, j] = transition_bias(
                        t, self.cat[i], self.tag[j], self.cat[j], biases)

    # ---------------------------------------------------------------- 표 출력

    def transition_table(self):
        """허용 전이 규칙을 사람이 눈으로 검증할 수 있는 표로 낸다."""
        groups = ["O", "B-X", "I-X", "E-X", "S-X"]

        def rep(g):
            if g == "O":
                return self.bg
            pre = g[0]
            for i in range(self.n):
                if self.tag[i] == pre:
                    return i
            return None

        rows = []
        for gi in groups:
            i = rep(gi)
            row = []
            for gj in groups:
                if gj == "O":
                    ok = self.trans[i, self.bg] > NEG_INF / 2
                else:
                    pre = gj[0]
                    cols = [j for j in range(self.n) if self.tag[j] == pre]
                    if self.cat[i] is None:
                        # 배경 O 는 카테고리가 없어 same/diff 구분이 무의미하다
                        ok = any(self.trans[i, j] > NEG_INF / 2 for j in cols)
                    else:
                        same = [j for j in cols if self.cat[j] == self.cat[i]]
                        diff = [j for j in cols if self.cat[j] != self.cat[i]]
                        s_ok = any(self.trans[i, j] > NEG_INF / 2 for j in same)
                        d_ok = any(self.trans[i, j] > NEG_INF / 2 for j in diff)
                        if s_ok and d_ok:
                            ok = "both"
                        elif s_ok:
                            ok = "same"
                        elif d_ok:
                            ok = "diff"
                        else:
                            ok = False
                row.append(ok)
            rows.append((gi, row))
        return groups, rows

    # ---------------------------------------------------------------- 디코딩

    def decode(self, logits):
        """logits [T,33] -> 라벨 id 리스트. 내부에서 log_softmax 를 적용한다.

        공식 decode() 가 token_logprobs 를 받으므로 로그확률로 맞춘다.
        """
        x = np.asarray(logits, dtype=np.float64)
        T = x.shape[0]
        if T == 0:
            return []
        m = x.max(axis=1, keepdims=True)
        lp = x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))

        score = lp[0] + self.start
        back = np.zeros((T - 1, self.n), dtype=np.int64)
        for t in range(1, T):
            cand = score[:, None] + self.trans      # [prev, next]
            best = cand.argmax(axis=0)
            back[t - 1] = best
            score = cand[best, np.arange(self.n)] + lp[t]

        score = score + self.end
        if not np.isfinite(score).any() or score.max() <= NEG_INF / 2:
            # 합법 경로가 없다 — 공식 구현과 같이 argmax 로 물러선다
            return lp.argmax(axis=1).tolist()

        path = [0] * T
        path[-1] = int(score.argmax())
        for t in range(T - 2, -1, -1):
            path[t] = int(back[t, path[t + 1]])
        return path


# ---------------------------------------------------------------- 스팬 변환

def spans_from_bioes(label_ids, offsets, id2label):
    """BIOES 를 그대로 해석한다. S-X 는 단일 토큰, B-X … E-X 가 하나의 스팬.

    group_spans 의 '연속 동일 카테고리 묶기' 를 쓰지 않는다.
    고아 I-/E- 는 스팬을 열지 않고 버린다 — 그게 1글자 스팬의 원인이었다.
    """
    spans = []
    cur = None                                   # (cat, start, end)
    for lid, (s, e) in zip(label_ids, offsets):
        if s == e:                               # 원문에 대응 없는 토큰
            continue
        t = id2label[str(lid)]
        if t == "O":
            cur = None
            continue
        pre, cat = t.split("-", 1)
        if pre == "S":
            spans.append({"label": cat, "start": s, "end": e})
            cur = None
        elif pre == "B":
            cur = [cat, s, e]
        elif pre == "I":
            if cur is not None and cur[0] == cat:
                cur[2] = e
            else:
                cur = None                       # 고아 I- 는 버린다
        elif pre == "E":
            if cur is not None and cur[0] == cat:
                spans.append({"label": cat, "start": cur[1], "end": e})
            cur = None                           # 고아 E- 는 버린다
    return spans


def count_illegal_transitions(label_ids, id2label):
    """라벨 시퀀스에 금지 전이가 몇 번 나오는지 센다. Viterbi 경로는 0 이어야 한다."""
    n = len(id2label)
    tag, cat, bg = parse_labels(id2label)
    bad = 0
    prev_t, prev_c = None, None
    first = True
    for lid in label_ids:
        t, c = tag[lid], cat[lid]
        if first:
            if not (t in ("B", "S") or lid == bg):
                bad += 1
            first = False
        else:
            if not is_valid_transition(prev_t, prev_c, t, c):
                bad += 1
        prev_t, prev_c = t, c
    if not first and not (prev_t in ("E", "S") or prev_t is None):
        bad += 1                                 # 열린 채로 끝났다
    return bad
