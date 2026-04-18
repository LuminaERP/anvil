"""Single source of truth for all embeddings across Anvil memory.

Previously the embedder config was duplicated across episodic.Memory and
SharedMemoryPool. This module centralises it so every part of the system
embeds into the same vector space — a precondition for comparing similarities
across lesson stores.

Upgraded in 2026 from `BAAI/bge-small-en-v1.5` to `Qwen/Qwen3-Embedding-0.6B`:
  - 32K context window (vs 512) — fits SWE-bench issues, long BCB specs, long
    LCB problem descriptions without truncation
  - Explicit code-retrieval training — better clustering for Anvil's mixed
    code + prose memory content
  - Matryoshka Representation Learning: we pick dim=384 at runtime to keep
    the existing vec0 index layout, with the option to step up to 1024 or
    1536 later without retraining

Configuration via env:
  AGENT_EMBEDDING_MODEL    — HF model id (default Qwen/Qwen3-Embedding-0.6B)
  AGENT_EMBEDDING_DIM      — Matryoshka dim (default 384)
  AGENT_EMBEDDING_DEVICE   — 'cuda' | 'cpu' | 'auto' (default 'auto')
  AGENT_EMBEDDING_PROMPT_NAME — optional prompt id for instruction-aware models

Backward compat: if the chosen model doesn't support MRL, we use its native
dim and log a warning. If the configured dim exceeds native, we pad with zeros.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---- configuration knobs (resolved once) -----------------------------------

DEFAULT_MODEL = os.environ.get("AGENT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_DIM = int(os.environ.get("AGENT_EMBEDDING_DIM", "384"))
DEFAULT_DEVICE = os.environ.get("AGENT_EMBEDDING_DEVICE", "auto")

# Legacy compatibility — the historical embedder was 384-dim BGE-small.
# vec0 virtual tables are created at 384 to match. New installs can go wider.
LEGACY_MODEL = "BAAI/bge-small-en-v1.5"
LEGACY_DIM = 384


# ---- singleton encoder ------------------------------------------------------

_MODEL: Optional["object"] = None
_LOAD_LOCK = threading.Lock()
_NATIVE_DIM: Optional[int] = None
_ACTUAL_DIM: Optional[int] = None
_ACTUAL_DEVICE: Optional[str] = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_model():
    """Lazy-load the SentenceTransformer model with the configured device."""
    global _MODEL, _NATIVE_DIM, _ACTUAL_DIM, _ACTUAL_DEVICE
    if _MODEL is not None:
        return _MODEL

    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        from sentence_transformers import SentenceTransformer  # type: ignore

        device = _resolve_device(DEFAULT_DEVICE)
        _ACTUAL_DEVICE = device

        try:
            model = SentenceTransformer(DEFAULT_MODEL, device=device, trust_remote_code=True)
        except TypeError:
            # Older sentence-transformers without trust_remote_code kwarg
            model = SentenceTransformer(DEFAULT_MODEL, device=device)

        # Probe native dimension
        try:
            _NATIVE_DIM = int(model.get_sentence_embedding_dimension())
        except Exception:
            _NATIVE_DIM = DEFAULT_DIM

        if DEFAULT_DIM <= _NATIVE_DIM:
            _ACTUAL_DIM = DEFAULT_DIM
        else:
            logger.warning(
                "configured embedding dim %d > model native dim %d; using native",
                DEFAULT_DIM, _NATIVE_DIM,
            )
            _ACTUAL_DIM = _NATIVE_DIM

        logger.info(
            "embedder loaded: model=%s device=%s native_dim=%d actual_dim=%d",
            DEFAULT_MODEL, device, _NATIVE_DIM, _ACTUAL_DIM,
        )
        _MODEL = model
        return _MODEL


# ---- public API -------------------------------------------------------------

def embed(text: str) -> bytes:
    """Embed a single text. Returns float32 bytes at ACTUAL_DIM.

    The byte layout matches the vec0 virtual-table expectation used by
    episodic.Memory and SharedMemoryPool — change this and those schemas
    break, so don't.
    """
    import numpy as np  # local import to avoid cycle during bootstrap

    model = _get_model()
    # Qwen3-Embedding and many recent models are instruction-aware. Using the
    # suggested retrieval prompt template for these yields measurable uplift,
    # at the cost of a small constant string prefix.
    prompt_text = _maybe_prefix_prompt(text)
    vec = model.encode(
        prompt_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # Matryoshka truncation
    if _ACTUAL_DIM is not None and vec.shape[-1] > _ACTUAL_DIM:
        vec = vec[..., :_ACTUAL_DIM]
        # Re-normalise after truncation so cosine stays comparable
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
    return np.asarray(vec, dtype=np.float32).tobytes()


def embed_batch(texts: list[str]) -> list[bytes]:
    """Batched variant — much faster when embedding many items at once."""
    import numpy as np
    if not texts:
        return []
    model = _get_model()
    prompts = [_maybe_prefix_prompt(t) for t in texts]
    arr = model.encode(
        prompts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    if _ACTUAL_DIM is not None and arr.shape[-1] > _ACTUAL_DIM:
        arr = arr[..., :_ACTUAL_DIM]
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
    return [np.asarray(row, dtype=np.float32).tobytes() for row in arr]


def info() -> dict:
    """Return current embedder state for diagnostics."""
    _get_model()  # force load to populate globals
    return {
        "model": DEFAULT_MODEL,
        "native_dim": _NATIVE_DIM,
        "actual_dim": _ACTUAL_DIM,
        "device": _ACTUAL_DEVICE,
    }


def embedding_dim() -> int:
    """Return the dimensionality the pool is using right now."""
    _get_model()
    return _ACTUAL_DIM or DEFAULT_DIM


# ---- helpers ----------------------------------------------------------------

def _is_qwen3_embedding(model_id: str) -> bool:
    return "qwen" in model_id.lower() and "embedding" in model_id.lower()


def _maybe_prefix_prompt(text: str) -> str:
    """Qwen3-Embedding models take a retrieval-query prompt for a small uplift.

    For non-Qwen3 models, pass text through unchanged so BGE / GTE / Nomic
    users see no behaviour change.
    """
    if _is_qwen3_embedding(DEFAULT_MODEL):
        # The official Qwen3-Embedding convention:
        #   Instruct: <task description>\nQuery: <text>
        # We use a generic coding-agent instruction that matches our workload.
        prompt = os.environ.get(
            "AGENT_EMBEDDING_PROMPT",
            "Given a software-engineering context or lesson, retrieve semantically similar ones.",
        )
        return f"Instruct: {prompt}\nQuery: {text}"
    return text
