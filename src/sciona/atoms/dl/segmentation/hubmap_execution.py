"""HuBMAP lifecycle bindings over explicit runtime arrays, encoder states and RNGs."""
from sciona.ghost.registry import register_atom


def witness_hubmap_initial_training(slides: object,groups: object,initial_stages: object,
                                    initial_selectors: object,tile_size: int=1024) -> tuple:
    return ({'kind':'HuBMAP.CheckpointEnsemble'},{'kind':'HuBMAP.TrainingMetrics'})


@register_atom(witness_hubmap_initial_training)
def hubmap_initial_training(slides: object,groups: object,initial_stages: object,
                            initial_selectors: object,tile_size: int=1024) -> tuple:
    """Prepare labeled folds and train fresh full networks with source losses."""
    from sciona.hubmap_graph_runtime import initial_training
    return initial_training(slides,groups,initial_stages,initial_selectors,tile_size)


def witness_hubmap_restore_models(checkpoints: object,input_resolution: int=320) -> dict:
    if checkpoints!={'kind':'HuBMAP.CheckpointEnsemble'}:
        raise ValueError('HuBMAP checkpoint ensemble required')
    return {'kind':'HuBMAP.InferenceEnsemble'}


@register_atom(witness_hubmap_restore_models)
def hubmap_restore_models(checkpoints: object,input_resolution: int=320) -> object:
    """Restore ordered plain-output full networks with batch classifier cutoff."""
    from sciona.hubmap_graph_runtime import restore_models
    return restore_models(checkpoints,input_resolution)


def witness_hubmap_pseudo_labels(models: object,pseudo_sources: object,pseudo_rng: object,
                                 inference_geometry: object=None) -> dict:
    if models!={'kind':'HuBMAP.InferenceEnsemble'}:
        raise ValueError('HuBMAP inference ensemble required')
    return {'kind':'HuBMAP.PseudoLabeledSources'}


@register_atom(witness_hubmap_pseudo_labels)
def hubmap_pseudo_labels(models: object,pseudo_sources: object,pseudo_rng: object,
                         inference_geometry: object=None) -> object:
    """Generate binary pseudo masks using source four-view TTA and stitching."""
    from sciona.hubmap_graph_runtime import pseudo_labels
    return pseudo_labels(models,pseudo_sources,pseudo_rng,inference_geometry)


def witness_hubmap_retraining(slides: object,groups: object,retraining_stages: object,
                              final_selectors: object,pseudo_labeled_sources: object,tile_size: int=1024) -> tuple:
    if pseudo_labeled_sources!={'kind':'HuBMAP.PseudoLabeledSources'}:
        raise ValueError('HuBMAP pseudo-labeled sources required')
    return ({'kind':'HuBMAP.CheckpointEnsemble'},{'kind':'HuBMAP.TrainingMetrics'})


@register_atom(witness_hubmap_retraining)
def hubmap_retraining(slides: object,groups: object,retraining_stages: object,
                      final_selectors: object,pseudo_labeled_sources: object,tile_size: int=1024) -> tuple:
    """Append positive pseudo tiles and train fresh encoder-initialized folds."""
    from sciona.hubmap_graph_runtime import retraining
    return retraining(slides,groups,retraining_stages,final_selectors,pseudo_labeled_sources,tile_size)


def witness_hubmap_final_prediction(models: object,final_images: object,final_rng: object,
                                    inference_geometry: object=None) -> dict:
    if models!={'kind':'HuBMAP.InferenceEnsemble'}:
        raise ValueError('HuBMAP inference ensemble required')
    return {'kind':'HuBMAP.BinaryPredictions'}


@register_atom(witness_hubmap_final_prediction)
def hubmap_final_prediction(models: object,final_images: object,final_rng: object,
                            inference_geometry: object=None) -> object:
    """Predict final binary masks using the notebook's three-view TTA."""
    from sciona.hubmap_graph_runtime import final_prediction
    return final_prediction(models,final_images,final_rng,inference_geometry)
