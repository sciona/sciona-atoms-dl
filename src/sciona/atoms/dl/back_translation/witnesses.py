"""Ghost witnesses for opaque back-translation atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractScalar


def witness_translate_text(
    text: AbstractScalar,
    src_lang: AbstractScalar,
    tgt_lang: AbstractScalar,
    model_path: AbstractScalar,
    model: AbstractScalar,
) -> AbstractScalar:
    """Witness a translation model producing one output string."""
    del text, src_lang, tgt_lang, model_path, model
    return AbstractScalar(dtype="str", min_val=1.0)
