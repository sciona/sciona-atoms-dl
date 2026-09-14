"""Draft Community Contrails execution boundaries; inputs supplied by caller."""
from sciona.ghost.registry import register_atom


def witness_contrails_prepare(payload: dict) -> dict:
    return {'kind': 'Contrails.Prepared'}


@register_atom(witness_contrails_prepare)
def contrails_prepare(payload: dict) -> object:
    """Validate versioned populations, numerical arrays and checkpoint policy."""
    from sciona.contrails_runtime import prepare
    return prepare(payload)


def witness_contrails_execute(prepared: object) -> dict:
    if prepared != {'kind': 'Contrails.Prepared'}:
        raise ValueError('Prepared Contrails input required')
    return {'kind': 'Contrails.Result'}


@register_atom(witness_contrails_execute)
def contrails_execute(prepared: object) -> dict:
    """Run all selected folds, terminal checkpoint inference, weighted ensemble and RLE."""
    from sciona.contrails_runtime import execute_prepared
    return execute_prepared(prepared)
