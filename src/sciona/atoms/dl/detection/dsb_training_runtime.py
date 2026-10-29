"""Draft explicit DSB model/optimizer runtime boundary for consecutive training."""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete shared model and optimizer state required')


@register_atom(concrete_only)
def restore_shared_training_runtime(initial_classifier_checkpoint: bytes, topk: int=5, start_epoch: int=0) -> object:
    """Restore source shared networks and two fresh SGD optimizers exactly once.

    This boundary is a model warm restart. The output retains shared parameters,
    optimizer momentum buffers and epoch origin throughout subsequent training.
    It does not restore historical optimizer or RNG state from checkpoint bytes.
    """
    from sciona.dsb_runtime import restore_training_runtime
    return restore_training_runtime(initial_classifier_checkpoint,topk=topk,start_epoch=start_epoch)


@register_atom(concrete_only)
def train_shared_runtime(runtime: object, training_factory: object, validation_factory: object,
                         end_epoch: int, topk: int=5, classifier_variant: int=3,
                         freeze_batchnorm: bool=False, prepare_epoch: object=None) -> tuple:
    """Run consecutive source epochs without recreating models, optimizers or RNG.

    Source validation and model-only checkpoint emission remain in this recipe
    loop. The final checkpoint and every epoch metric are retained explicitly.
    Loss, validation and checkpoint operations still require separate review.
    """
    from sciona.dsb_training_range import run_training_range
    if getattr(training_factory,'topk',None)!=topk:
        raise ValueError('Matching training factory topk required')
    if runtime['classifier'].NoduleNet is not runtime['detector']:
        raise ValueError('Shared detector/classifier feature network required')
    for role in ('detector','classifier'):
        expected={id(p) for p in runtime[role].parameters()}
        actual={id(p) for group in runtime[role+'_optimizer'].param_groups for p in group['params']}
        if actual!=expected:raise ValueError('Optimizer parameter ownership differs')
    result=run_training_range(runtime,training_factory,validation_factory,end_epoch=end_epoch,
        classifier_variant=classifier_variant,freeze_batchnorm=freeze_batchnorm,prepare_epoch=prepare_epoch)
    return result['checkpoint'],result['epochs']
