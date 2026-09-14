"""Candidate DSB runtime bindings; registration does not confer catalog approval."""
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_restore_dsb_models(detector_checkpoint: object, classifier_checkpoint: object,
                               topk: int=5) -> tuple:
    if isinstance(topk,bool) or not isinstance(topk,int) or topk<1:
        raise ValueError('positive integer topk required')
    # Checkpoint contents can only be validated by the runtime loader.
    return ({'model':'DSB.DetectorNet','mode':'eval'},
            {'model':'DSB.CaseNet','mode':'eval','topk':topk})


@register_atom(witness_restore_dsb_models)
def restore_dsb_models(detector_checkpoint: bytes, classifier_checkpoint: bytes,
                       topk: int=5) -> tuple:
    """Restore separate, explicit metadata-free detector and classifier states."""
    from sciona.dsb_runtime import restore_inference_runtime
    models=restore_inference_runtime(detector_checkpoint,classifier_checkpoint,topk=topk)
    return models['detector'],models['classifier']


def witness_dsb_preprocess(volume: AbstractArray, spacing: object,
                           requested_spacing: object=(1.,1.,1.)) -> AbstractArray:
    if len(volume.shape)!=3 or volume.shape[1]!=volume.shape[2]:
        raise ValueError('source segmentation requires a volume with square planes')
    # Segmentation and crop bounds depend on values, not shape alone.
    return AbstractArray(shape=(1,-1,-1,-1),dtype='uint8',min_val=0,max_val=255)


@register_atom(witness_dsb_preprocess)
def dsb_preprocess(volume: object, spacing: object,
                   requested_spacing: object=(1.,1.,1.)) -> object:
    """Segment lungs, mask bone, resample and crop using pinned source semantics."""
    from sciona.dsb_components import preprocess_volume
    processed,_,_=preprocess_volume(volume,spacing,requested_spacing)
    return processed[None]


def witness_dsb_predict(processed_volume: AbstractArray, detector: dict, classifier: dict,
                        topk: int=5, crop_size: int=96, tile_side: int=144,
                        margin: int=32, tile_batch_size: int=1) -> tuple:
    if len(processed_volume.shape)!=4 or processed_volume.shape[0]!=1:
        raise ValueError('preprocessed CZYX volume required')
    if detector.get('model')!='DSB.DetectorNet' or detector.get('mode')!='eval':
        raise ValueError('eval detector required')
    if classifier!={'model':'DSB.CaseNet','mode':'eval','topk':topk}:
        raise ValueError('matching eval classifier required')
    for name,value in [('topk',topk),('crop_size',crop_size),('tile_side',tile_side),('tile_batch_size',tile_batch_size)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(name+' must be a positive integer')
    if crop_size%16 or tile_side%16 or isinstance(margin,bool) or not isinstance(margin,int) or margin<0 or margin%16:
        raise ValueError('network geometry requires multiples of sixteen')
    return (AbstractArray(shape=(1,),dtype='float32',min_val=0,max_val=1),
            AbstractArray(shape=(1,topk),dtype='float32',min_val=0,max_val=1))


@register_atom(witness_dsb_predict)
def dsb_predict(processed_volume: object, detector: object, classifier: object,
                topk: int=5, crop_size: int=96, tile_side: int=144,
                margin: int=32, tile_batch_size: int=1) -> tuple:
    """Execute split/combine detector, proposal selection, crop and noisy-OR head.

    This inference branch excludes training, oversampling, target assignment and
    training losses. Caller-provided states do not imply predictive quality.
    """
    from sciona.dsb_pipeline import infer_preprocessed
    result=infer_preprocessed(processed_volume,detector,classifier,topk=topk,
        crop_size=crop_size,tile_side=tile_side,margin=margin,tile_batch_size=tile_batch_size)
    return result['case_probability'],result['proposal_probabilities']
