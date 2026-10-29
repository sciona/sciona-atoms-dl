"""Draft recipe bindings for checkpoint-compatible reusable classifier pooling."""
from sciona.ghost.registry import register_atom
from sciona.atoms.dl.detection.dsb_training_runtime import concrete_only


@register_atom(concrete_only)
def restore_shared_training_runtime(initial_classifier_checkpoint: bytes,topk: int=5,start_epoch: int=0) -> object:
    """Restore shared networks and fresh optimizers with explicit noisy-OR pooling."""
    from sciona.dsb_decomposed_model_runtime import restore_training_runtime
    return restore_training_runtime(initial_classifier_checkpoint,topk=topk,start_epoch=start_epoch)


@register_atom(concrete_only)
def restore_inference_models(detector_checkpoint: bytes,classifier_checkpoint: bytes,topk: int=5) -> tuple:
    """Restore separate explicit eval states with reusable classifier pooling."""
    from sciona.dsb_decomposed_model_runtime import restore_inference_runtime
    runtime=restore_inference_runtime(detector_checkpoint,classifier_checkpoint,topk=topk)
    return runtime['detector'],runtime['classifier']
