"""Draft DSB stateful training recipe using a reusable classifier penalty."""
from sciona.ghost.registry import register_atom
from sciona.atoms.dl.detection.dsb_training_runtime import concrete_only


@register_atom(concrete_only)
def train_shared_runtime(runtime: object, training_factory: object, validation_factory: object,
                         end_epoch: int, topk: int=5, classifier_variant: int=3,
                         freeze_batchnorm: bool=False, prepare_epoch: object=None) -> tuple:
    """Run consecutive source epochs without recreating models, optimizers or RNG.

    Source validation and model-only checkpoint emission remain in this recipe
    loop. The final checkpoint and every epoch metric are retained explicitly.
    Loss, validation and checkpoint operations still require separate review.
    """
    from sciona.dsb_decomposed_lifecycle import run_training_range
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
