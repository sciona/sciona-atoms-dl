"""Source recipe boundary accepting a graph-selected numerical operation."""
from sciona.ghost.registry import register_atom


def witness_execute_with_normalizer(*args, **kwargs):
    raise ValueError('Concrete prepared attack and row-normalization callable required')


@register_atom(witness_execute_with_normalizer)
def execute_with_normalizer(prepared: object, normalizer: object) -> dict:
    """Run the complete padded attack with an explicitly supplied row normalizer.

    The source recipe chooses mean-absolute or population-standard-deviation
    scaling at each recurrent update. The graph supplies the numerical function.
    Preserve all source model branches, label rules, momentum updates, projection,
    and real-entry PNG encoding. No implicit pretrained state is downloaded.
    """
    from sciona.decomposed_adversarial_injected import execute
    return execute(prepared, normalizer)
