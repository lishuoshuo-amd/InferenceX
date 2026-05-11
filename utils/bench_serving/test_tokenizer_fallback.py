"""Regression tests for the tokenizer loader fallback in benchmark_serving.

Older vLLM (< v0.12.0, before upstream PR #29686) accesses
``all_special_tokens_extended`` which is missing on some HF tokenizers
(e.g. Qwen2/Qwen3), crashing benchmark setup. The wrapper ``get_tokenizer``
in benchmark_serving falls back to the HF AutoTokenizer path only for that
specific AttributeError. These tests pin that contract.
"""
import sys
from pathlib import Path

import pytest

# Ensure ``import benchmark_serving`` resolves to the sibling module.
sys.path.insert(0, str(Path(__file__).parent))

# Module imports heavy deps (numpy, transformers, etc.); skip cleanly if absent.
bs = pytest.importorskip(
    "benchmark_serving",
    reason="benchmark_serving requires numpy/transformers; skip in lite envs",
)


@pytest.fixture(autouse=True)
def _reset_module(monkeypatch):
    """Ensure each test sees a fresh state for the patched module attrs."""
    monkeypatch.setattr(bs, "_vllm_get_tokenizer", None, raising=False)
    monkeypatch.setattr(bs, "_hf_get_tokenizer", lambda *a, **kw: None, raising=False)
    yield


def test_uses_vllm_when_available(monkeypatch):
    """Healthy vLLM path: wrapper returns vLLM tokenizer, never calls HF."""
    sentinel = object()
    hf_called = {"n": 0}

    def fake_vllm(*a, **kw):
        return sentinel

    def fake_hf(*a, **kw):
        hf_called["n"] += 1
        return "hf-tokenizer"

    monkeypatch.setattr(bs, "_vllm_get_tokenizer", fake_vllm)
    monkeypatch.setattr(bs, "_hf_get_tokenizer", fake_hf)

    result = bs.get_tokenizer("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")

    assert result is sentinel
    assert hf_called["n"] == 0


def test_falls_back_on_all_special_tokens_extended(monkeypatch):
    """Legacy vLLM path: AttributeError on all_special_tokens_extended -> HF."""
    def fake_vllm(*a, **kw):
        raise AttributeError(
            "'Qwen2TokenizerFast' object has no attribute "
            "'all_special_tokens_extended'"
        )

    def fake_hf(*a, **kw):
        return "hf-tokenizer"

    monkeypatch.setattr(bs, "_vllm_get_tokenizer", fake_vllm)
    monkeypatch.setattr(bs, "_hf_get_tokenizer", fake_hf)

    result = bs.get_tokenizer(
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    assert result == "hf-tokenizer"


def test_reraises_other_attribute_errors(monkeypatch):
    """Non-target AttributeError must propagate, not be swallowed."""
    def fake_vllm(*a, **kw):
        raise AttributeError("some other unrelated attribute")

    def fake_hf(*a, **kw):
        pytest.fail("HF fallback must not be invoked for unrelated errors")

    monkeypatch.setattr(bs, "_vllm_get_tokenizer", fake_vllm)
    monkeypatch.setattr(bs, "_hf_get_tokenizer", fake_hf)

    with pytest.raises(AttributeError, match="unrelated attribute"):
        bs.get_tokenizer("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")


def test_uses_hf_when_vllm_unavailable(monkeypatch):
    """When vLLM is not importable, wrapper goes straight to HF loader."""
    hf_called = {"n": 0}

    def fake_hf(*a, **kw):
        hf_called["n"] += 1
        return "hf-tokenizer"

    monkeypatch.setattr(bs, "_vllm_get_tokenizer", None)
    monkeypatch.setattr(bs, "_hf_get_tokenizer", fake_hf)

    result = bs.get_tokenizer("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")

    assert result == "hf-tokenizer"
    assert hf_called["n"] == 1
