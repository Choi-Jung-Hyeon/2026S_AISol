from typing import List, Tuple, Dict, Any


def split_tag(tag: str) -> Tuple[str, str]:
    """'B-private_person' -> ('B', 'private_person'),  'O' -> ('O', '')"""
    if tag == "O" or "-" not in tag:
        return "O", ""
    prefix, label = tag.split("-", 1)
    return prefix, label


def group_spans(
    tags: List[str],
    offsets: List[Tuple[int, int]],
    calib: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    연속된 같은 라벨을 하나의 스팬으로 묶는 작업
    calib 은 Viterbi 시그니처를 맞추기 위한 것으로 A안에서는 쓰지 않음
    """
    spans: List[Dict[str, Any]] = []
    cur_label = cur_start = cur_end = None

    def close():
        nonlocal cur_label, cur_start, cur_end
        if cur_label is not None:
            spans.append({"label": cur_label, "start": cur_start, "end": cur_end})
        cur_label = cur_start = cur_end = None

    for tag, (s, e) in zip(tags, offsets):
        if s == e:                          # 특수토큰 등 원문에 대응 없는 토큰
            continue

        prefix, label = split_tag(tag)

        if prefix == "O":
            close()

        elif prefix == "S":
            close()
            spans.append({"label": label, "start": s, "end": e})

        elif prefix == "B":
            close()
            cur_label, cur_start, cur_end = label, s, e

        elif prefix in ("I", "E"):
            if cur_label == label:
                cur_end = e
            else:                           # 시작 없이 중간/끝이 나온 경우
                close()
                cur_label, cur_start, cur_end = label, s, e
            if prefix == "E":
                close()

    close()
    return spans


def mask_text(text: str, spans: List[Dict[str, Any]]) -> str:
    """스팬 위치를 [라벨명] 으로 치환. 뒤에서부터 바꿔야 위치가 안 밀림"""
    out = text
    for sp in sorted(spans, key=lambda x: -x["start"]):
        out = out[:sp["start"]] + f"[{sp['label']}]" + out[sp["end"]:]
    return out
