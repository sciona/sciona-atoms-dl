"""DSB preparation bindings with explicit reusable resampling dependency."""
from sciona.ghost.registry import register_atom
from sciona.atoms.dl.detection.dsb_training_execution import witness_dsb_raw_collections
from sciona.atoms.dl.detection.dsb_execution import witness_dsb_preprocess


@register_atom(witness_dsb_raw_collections)
def prepare_raw_collections(raw_collections: dict, eligible_image_indices: object) -> tuple:
    """Prepare five caller-supplied populations under explicit coordinate policies."""
    from sciona._runtime.dl.dsb_decomposed_raw_inputs import prepare_raw_collections as prepare
    return prepare(raw_collections, eligible_image_indices)


@register_atom(witness_dsb_preprocess)
def preprocess_volume(volume: object, spacing: object,
                      requested_spacing: object=(1.,1.,1.)) -> object:
    """Apply source inference segmentation, cleanup, generic resampling and crop."""
    from sciona._runtime.dl.dsb_decomposed_preparation import preprocess_volume as prepare
    processed, _, _ = prepare(volume, spacing, requested_spacing)
    return processed[None]
