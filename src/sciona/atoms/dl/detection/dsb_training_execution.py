"""Candidate in-memory DSB training bindings; no data or states are embedded."""
from sciona.ghost.registry import register_atom


def witness_dsb_raw_collections(raw_collections: dict,eligible_image_indices: object) -> tuple:
    if set(raw_collections)!={'training_detector','training_classifier','validation_detector',
                              'validation_classifier','training_validation_classifier'}:
        raise ValueError('five explicit raw collections required')
    return ({name:None for name in ['detector_volumes','boxes_by_image','eligible_image_indices',
            'classifier_volumes','proposals_by_case','known_by_case','case_labels']},
            {name:None for name in ['detector','classifier_validation','classifier_training_validation']})


@register_atom(witness_dsb_raw_collections)
def dsb_raw_collections(raw_collections: dict,eligible_image_indices: object) -> tuple:
    """Prepare raw volumes/annotations and label training proposals in memory."""
    from sciona.dsb_raw_inputs import prepare_raw_collections
    return prepare_raw_collections(raw_collections,eligible_image_indices)


def witness_dsb_adapt_detector(detector_checkpoint: object,topk: int=5) -> dict:
    if isinstance(topk,bool) or not isinstance(topk,int) or topk<1:
        raise ValueError('positive integer topk required')
    return {'kind':'DSB.CaseNetCheckpoint','epoch':0}


@register_atom(witness_dsb_adapt_detector)
def dsb_adapt_detector(detector_checkpoint: bytes,topk: int=5) -> bytes:
    """Initialize the shared classifier from an explicit detector checkpoint."""
    from sciona.dsb_checkpoint_adaptation import adapt_detector_checkpoint
    return adapt_detector_checkpoint(detector_checkpoint,topk=topk)


def witness_dsb_validation_batches(validation_arrays: dict,validation_options: dict) -> dict:
    if set(validation_arrays)!={'detector','classifier_validation','classifier_training_validation'}:
        raise ValueError('three explicit validation collections required')
    return {'kind':'DSB.ValidationBatchFactory','topk':validation_options.get('topk',5)}


@register_atom(witness_dsb_validation_batches)
def dsb_validation_batches(validation_arrays: dict,validation_options: dict) -> object:
    """Construct source validation batches from runtime arrays and annotations."""
    from sciona.dsb_validation_batches import ValidationBatchFactory
    witness_dsb_validation_batches(validation_arrays,validation_options)
    return ValidationBatchFactory(validation_arrays,**validation_options)


def witness_dsb_training_batches(training_arrays: dict, sampling_options: dict) -> dict:
    required={'detector_volumes','boxes_by_image','eligible_image_indices',
              'classifier_volumes','proposals_by_case','known_by_case','case_labels'}
    if set(training_arrays)!=required:
        raise ValueError('explicit aligned training collections required')
    if not {'detector_order','classifier_order','detector_batch_size','classifier_batch_size'}<=set(sampling_options):
        raise ValueError('explicit sample orders and batch sizes required')
    return {'kind':'DSB.TrainingBatchFactory','topk':sampling_options.get('topk',5)}


@register_atom(witness_dsb_training_batches)
def dsb_training_batches(training_arrays: dict, sampling_options: dict) -> object:
    """Connect runtime arrays to size sampling, crops, augmentation and labels.

    Arrays must already be source-preprocessed. Caller supplies complete epoch
    permutations and owns RNG streams; historical worker scheduling is excluded.
    """
    from sciona.dsb_batches import TrainingBatchFactory
    witness_dsb_training_batches(training_arrays,sampling_options)
    return TrainingBatchFactory(**training_arrays,**sampling_options)


def witness_dsb_train_epoch(initial_classifier_checkpoint: object, training_factory: dict,
                            validation_factory: object, epoch: int, start_epoch: int=0,
                            topk: int=5, classifier_variant: int=3,
                            freeze_batchnorm: bool=False) -> tuple:
    if training_factory!={'kind':'DSB.TrainingBatchFactory','topk':topk}:
        raise ValueError('matching DSB training factory required')
    if classifier_variant not in (3,4) or isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<1:
        raise ValueError('valid epoch and classifier variant required')
    return ({'kind':'DSB.CaseNetCheckpoint','epoch':epoch},
            {'kind':'DSB.EpochAggregateMetrics'})


@register_atom(witness_dsb_train_epoch)
def dsb_train_epoch(initial_classifier_checkpoint: bytes, training_factory: object,
                    validation_factory: object, epoch: int, start_epoch: int=0,
                    topk: int=5, classifier_variant: int=3,
                    freeze_batchnorm: bool=False) -> tuple:
    """Warm-restart one source-ordered epoch and return its checkpoint and metrics.

    Includes initial zero-LR pass when appropriate, detector OHEM and classifier
    miss penalty, validation and model-only checkpoint. This node is a warm-restart
    boundary; do not chain it to claim uninterrupted multi-epoch optimizer state.
    """
    from sciona.dsb_runtime import restore_training_runtime
    from sciona.dsb_training import run_epoch_lifecycle
    if getattr(training_factory,'topk',None)!=topk:
        raise ValueError('matching DSB training factory required')
    runtime=restore_training_runtime(initial_classifier_checkpoint,topk=topk,start_epoch=start_epoch)
    result=run_epoch_lifecycle(**runtime,training_factory=training_factory,
        validation_factory=validation_factory,epoch=epoch,classifier_variant=classifier_variant,
        freeze_batchnorm=freeze_batchnorm,save_frequency=1)
    return result['checkpoint'],{'training':result['training'],'validation':result['validation']}


def witness_dsb_train_range(initial_classifier_checkpoint: object,training_factory: dict,
                            validation_factory: object,end_epoch: int,start_epoch: int=0,
                            topk: int=5,classifier_variant: int=3,
                            freeze_batchnorm: bool=False,prepare_epoch: object=None) -> tuple:
    checkpoint,_=witness_dsb_train_epoch(initial_classifier_checkpoint,training_factory,
        validation_factory,end_epoch,start_epoch,topk,classifier_variant,freeze_batchnorm)
    return checkpoint,{'kind':'DSB.EpochRangeAggregateMetrics'}


@register_atom(witness_dsb_train_range)
def dsb_train_range(initial_classifier_checkpoint: bytes,training_factory: object,
                     validation_factory: object,end_epoch: int,start_epoch: int=0,
                     topk: int=5,classifier_variant: int=3,
                     freeze_batchnorm: bool=False,prepare_epoch: object=None) -> tuple:
    """Restore once, train consecutive epochs with persistent optimizers, export final state."""
    from sciona.dsb_runtime import restore_training_runtime
    from sciona.dsb_training_range import run_training_range
    if getattr(training_factory,'topk',None)!=topk:
        raise ValueError('matching DSB training factory required')
    runtime=restore_training_runtime(initial_classifier_checkpoint,topk=topk,start_epoch=start_epoch)
    result=run_training_range(runtime,training_factory,validation_factory,end_epoch=end_epoch,
        classifier_variant=classifier_variant,freeze_batchnorm=freeze_batchnorm,prepare_epoch=prepare_epoch)
    return result['checkpoint'],result['epochs']
