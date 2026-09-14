"""Draft DFDC Community execution boundaries with explicit caller inputs."""
from sciona.ghost.registry import register_atom


def witness_dfdc_prepare(payload: dict) -> dict:
    return {'kind':'DFDC.Prepared'}


@register_atom(witness_dfdc_prepare)
def dfdc_prepare(payload: dict) -> object:
    """Validate versioned clips, states, pair folds and explicit checkpoint plan."""
    from sciona.dfdc_runtime import prepare
    return prepare(payload)


def witness_dfdc_execute(prepared: object) -> dict:
    if prepared != {'kind':'DFDC.Prepared'}:
        raise ValueError('Prepared DFDC input required')
    return {'kind':'DFDC.Result'}


@register_atom(witness_dfdc_execute)
def dfdc_execute(prepared: object) -> dict:
    """Run actual crop/annotation preparation, B7 runs and selected ensemble inference."""
    from sciona.dfdc_runtime import execute_prepared
    return execute_prepared(prepared)
