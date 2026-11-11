"""Explicit DSB validation and model-only checkpoint contracts.

Validation changes the supplied model to eval mode, consumes the supplied batch
iterator and returns aggregate metrics. Epoch scheduling remains caller policy.
Checkpoint export snapshots model tensors only: optimizer and random states are
intentionally absent, so these bytes are a warm-restart boundary.
"""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete model and runtime batches required')


@register_atom(concrete_only)
def validate_detector(model: object, batches: object) -> dict:
    """Evaluate detector batches; average batch losses equally and sum counts.

    Batches contain images, anchor labels and coordinate channels. Requires a
    nonempty iterable. Sets eval mode and disables gradients. Does not update
    parameters or optimizers; returning to training mode is caller policy.
    """
    from sciona._runtime.dl.dsb_decomposed_validation import validate_detector as evaluate
    return evaluate(model, batches)


@register_atom(concrete_only)
def validate_classifier(model: object, batches: object) -> dict:
    """Evaluate classifier batches using source case-weighted losses and accuracy.

    Batches contain images, coordinates, known-proposal flags and case labels.
    Requires a nonempty iterable. Sets eval mode and disables gradients; strict
    probability > .5 determines predicted case labels. Returns aggregates only.
    """
    from sciona._runtime.dl.dsb_decomposed_validation import validate_classifier as evaluate
    return evaluate(model, batches)


@register_atom(concrete_only)
def export_model_checkpoint(model: object, epoch: int) -> bytes:
    """Copy model tensors to CPU and serialize only epoch and state_dict.

    Epoch must be a nonnegative integer. No optimizer state, RNG state, runtime
    options or input arrays are serialized. Model device, mode, tensors and
    live optimizer state remain unchanged. The caller controls byte storage.
    """
    from sciona._runtime.dl.dsb_checkpoint import checkpoint_bytes
    return checkpoint_bytes(model, epoch)
