from __future__ import annotations

from pathlib import Path

import pytest

from sciona.atoms.dl.back_translation import translate_text


class FakeTranslationModel:
    def __call__(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return f"{tgt_lang}:{text[::-1]}:{src_lang}"


class EmptyTranslationModel:
    def __call__(self, text: str, src_lang: str, tgt_lang: str) -> str:
        del text, src_lang, tgt_lang
        return " "


def test_translate_text_uses_injected_backend_without_loading_models() -> None:
    translated = translate_text(
        "This is a short toxic-comment fixture.",
        "en",
        "fr",
        Path("Helsinki-NLP/opus-mt-en-fr"),
        FakeTranslationModel(),
    )

    assert translated == "fr:.erutxif tnemmoc-cixot trohs a si sihT:en"


def test_translate_text_supports_back_translation_composition() -> None:
    forward = translate_text(
        "clean input",
        "en",
        "de",
        "Helsinki-NLP/opus-mt-en-de",
        FakeTranslationModel(),
    )
    backward = translate_text(
        forward,
        "de",
        "en",
        "Helsinki-NLP/opus-mt-de-en",
        FakeTranslationModel(),
    )

    assert backward.startswith("en:")
    assert backward.endswith(":de")


def test_translate_text_rejects_bad_contract_inputs() -> None:
    with pytest.raises(Exception, match="non-empty"):
        translate_text("", "en", "fr", "model", FakeTranslationModel())

    with pytest.raises(Exception, match="differ"):
        translate_text("hello", "en", "en", "model", FakeTranslationModel())

    with pytest.raises(Exception, match="empty text"):
        translate_text("hello", "en", "fr", "model", EmptyTranslationModel())
