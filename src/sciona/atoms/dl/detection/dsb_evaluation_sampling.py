"""DSB validation and inference bindings with explicit spatial recipe dependencies."""
from sciona.ghost.registry import register_atom
from sciona.atoms.dl.detection.dsb_training_execution import witness_dsb_validation_batches
from sciona.atoms.dl.detection.dsb_inference_steps import concrete_only


@register_atom(witness_dsb_validation_batches)
def validation_batches(validation_arrays: dict, validation_options: dict) -> object:
    """Construct three source validation populations with caller-owned random streams."""
    from sciona.dsb_decomposed_validation_batches import ValidationBatchFactory
    witness_dsb_validation_batches(validation_arrays, validation_options)
    return ValidationBatchFactory(validation_arrays, **validation_options)


@register_atom(concrete_only)
def prepare_classifier_crops(processed_volume: object, proposals: object,
                             topk: int=5, crop_size: int=96) -> tuple:
    """Build source score-ordered test crops, retaining zero slots and padded coordinates."""
    from sciona.dsb_decomposed_evaluation_samples import prepare_classifier_batch
    crops, coordinates, _ = prepare_classifier_batch(processed_volume, proposals, topk, crop_size)
    return crops, coordinates
