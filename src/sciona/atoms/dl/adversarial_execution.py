"""Draft Community boundaries for explicit source momentum-attack execution."""
from sciona.ghost.registry import register_atom


def witness_adversarial_prepare(payload: dict) -> dict:
    return {'kind':'Adversarial.Prepared'}


@register_atom(witness_adversarial_prepare)
def adversarial_prepare(payload: dict) -> object:
    """Decode versioned private image/target/state payloads with explicit configuration."""
    from sciona.adversarial_runtime import prepare
    return prepare(payload)


def witness_adversarial_execute(prepared: object) -> dict:
    if prepared != {'kind':'Adversarial.Prepared'}:
        raise ValueError('Prepared adversarial input required')
    return {'kind':'Adversarial.Result'}


@register_atom(witness_adversarial_execute)
def adversarial_execute(prepared: object) -> dict:
    """Run complete padded source ensemble attack and encode real-entry outputs."""
    from sciona.adversarial_runtime import execute
    return execute(prepared)
