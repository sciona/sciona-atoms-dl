"""Draft DSB recipe bindings; registration does not confer catalog approval."""
from sciona.ghost.registry import register_atom
from sciona.atoms.dl.detection.dsb_training_execution import witness_dsb_training_batches


@register_atom(witness_dsb_training_batches)
def dsb_decomposed_training_batches(training_arrays: dict, sampling_options: dict) -> object:
    """Build source-ordered batches using reusable repetition and score sampling.

    This is a DSB recipe adapter with explicit arrays, orders and RNG streams.
    Cropping, augmentation, labels and model conventions remain source-specific.
    The factory preserves random stream continuity across passes and epochs.
    """
    from sciona.dsb_decomposed_batches import TrainingBatchFactory
    witness_dsb_training_batches(training_arrays, sampling_options)
    return TrainingBatchFactory(**training_arrays, **sampling_options)
