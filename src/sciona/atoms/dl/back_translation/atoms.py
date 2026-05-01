"""Opaque translation atom for back-translation augmentation pipelines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_translate_text


def _lang_valid(value: str) -> bool:
    return bool(value.strip()) and value.strip().replace("-", "").isalpha()


def _model_path_valid(model_path: str | Path) -> bool:
    return bool(str(model_path).strip())


@register_atom(witness_translate_text)
@icontract.require(lambda text: bool(text.strip()), "text must be non-empty")
@icontract.require(lambda src_lang: _lang_valid(src_lang), "src_lang must be a language code")
@icontract.require(lambda tgt_lang: _lang_valid(tgt_lang), "tgt_lang must be a language code")
@icontract.require(lambda src_lang, tgt_lang: src_lang != tgt_lang, "source and target languages must differ")
@icontract.require(lambda model_path: _model_path_valid(model_path), "model_path must identify the frozen model")
@icontract.require(lambda model: callable(model), "model must be a callable translation backend")
@icontract.ensure(lambda result: bool(result.strip()), "translated text must be non-empty")
def translate_text(
    text: str,
    src_lang: str,
    tgt_lang: str,
    model_path: str | Path,
    model: Callable[[str, str, str], str],
) -> str:
    """Translate text with an injected pretrained translation model.

    The atom does not download weights, load HuggingFace caches, or move models
    between devices. Those side effects belong to the orchestrator. This
    function only executes the already-loaded model's deterministic translation
    boundary and returns its text output.
    """
    del model_path
    translated = model(text, src_lang, tgt_lang)
    if not isinstance(translated, str):
        raise TypeError("translation backend must return a string")
    if not translated.strip():
        raise ValueError("translation backend returned empty text")
    return translated
