"""OPF 로컬 추론 어댑터.

    입력 : str
    출력 : (spans, masked_text)

후처리(group_spans / mask_text)는 postproc.py 를 그대로 import 한다.
여기서 다시 구현하지 않는다 — 후처리가 갈리면 기존 결과와 비교가 무의미해진다.

원래 진입점과 맞춘 지점
  - tokenizer.encode(text) 를 기본 인자로 호출한다 (add_special_tokens 를 끄지 않는다).
  - group_spans(tags, enc.offsets) 로 인자 2개만 넘긴다.
  - logits.shape[0] != len(enc.offsets) 이면 즉시 sys.exit 한다.
    토크나이저가 갈리면 여기서 걸린다.
"""

from __future__ import annotations

import json
import os
import sys

from postproc import group_spans, mask_text

DEFAULT_CKPT = os.path.expanduser("~/.opf/privacy_filter")


def load_id2label(ckpt=DEFAULT_CKPT):
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
        return json.load(f)["id2label"]


def load_tokenizer(ckpt=DEFAULT_CKPT):
    from tokenizers import Tokenizer

    path = os.path.join(ckpt, "tokenizer.json")
    if not os.path.isfile(path):
        sys.exit("tokenizer.json 이 없습니다: %s" % path)
    return Tokenizer.from_file(path)


class LocalBackend:
    """이 장비에서 직접 로짓 [T,33] 을 낸다."""

    name = "local"

    def __init__(self, ckpt=DEFAULT_CKPT, device=None, dtype=None):
        import torch
        from transformers import AutoModelForTokenClassification

        if device is None:
            # 실측상 이 장비에서는 CPU 가 MPS 보다 6배 빠르다 (건당 0.173초 vs 1.062초).
            # MoE 128 experts top-4 의 게더 연산이 잘아서 MPS 로 못 넘긴다.
            device = "cpu"
        if dtype is None:
            # bfloat16 은 mps/cpu 에서 느리거나 미지원이라 float32 로 올린다.
            dtype = torch.float32
        self.torch = torch
        self.device = device
        self.model = AutoModelForTokenClassification.from_pretrained(ckpt, dtype=dtype)
        self.model.eval()
        self.model.to(device)

    def logits(self, token_ids):
        torch = self.torch
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=attn)
        return out.logits[0].float().cpu().numpy()



def tag_tokens(text, tokenizer, backend, id2label):
    """토큰별 (문자열, 오프셋, 예측태그) 를 낸다. 후처리에 의존하지 않는다.

    토큰 수가 어긋나면 즉시 중단한다.
    """
    enc = tokenizer.encode(text)
    logits = backend.logits(enc.ids)
    if logits.shape[0] != len(enc.offsets):
        sys.exit(
            "토큰 수 불일치: %s=%d, 로컬=%d — tokenizer.json 이 서로 다르다"
            % (backend.name, logits.shape[0], len(enc.offsets))
        )
    tags = [id2label[str(i)] for i in logits.argmax(-1)]
    return enc, tags, logits


def redact(text, tokenizer, backend, id2label):
    """str -> (spans, masked_text)."""
    enc, tags, _ = tag_tokens(text, tokenizer, backend, id2label)
    spans = group_spans(tags, enc.offsets)
    return spans, mask_text(text, spans)


def build(backend="local", ckpt=DEFAULT_CKPT, device=None):
    tokenizer = load_tokenizer(ckpt)
    id2label = load_id2label(ckpt)
    return tokenizer, LocalBackend(ckpt, device), id2label
