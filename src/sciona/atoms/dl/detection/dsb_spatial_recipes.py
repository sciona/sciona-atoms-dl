"""Explicit DSB crop, augmentation and anchor recipes with caller-owned RNG.

These are source-policy adapters, not domain-independent geometry claims.
Detector flips use size-coordinate; classifier rotation leaves coordinates
unchanged; classifier crops use the padded coordinate frame.
"""
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete arrays and explicit random streams required')


@register_atom(concrete_only)
def crop_detection(volume: object, target: object, boxes: object, *, crop_size: object=128, bound_size: object=12, stride: object=4, scale: object=False, random_crop: object=False, pad_value: object=170, rng: object=None) -> object:
    """Source training crop and transformed labels, using an explicit shared RNG.

An explicit RNG is required; the caller retains and advances its state."""
    from sciona._runtime.dl.dsb_detector_crop import crop_detection as implementation
    if rng is None:
        raise ValueError('Explicit caller-owned RNG required')
    return implementation(volume=volume, target=target, boxes=boxes, crop_size=crop_size, bound_size=bound_size, stride=stride, scale=scale, random_crop=random_crop, pad_value=pad_value, rng=rng)

@register_atom(concrete_only)
def augment_detection(sample: object, target: object, boxes: object, coord: object, *, flip: object=True, rotate: object=True, swap: object=True, rng: object=None) -> object:
    """augment_detection

An explicit RNG is required; the caller retains and advances its state."""
    from sciona._runtime.dl.dsb_augmentation import augment_detection as implementation
    if rng is None:
        raise ValueError('Explicit caller-owned RNG required')
    return implementation(sample=sample, target=target, boxes=boxes, coord=coord, flip=flip, rotate=rotate, swap=swap, rng=rng)

@register_atom(concrete_only)
def augment_classifier(sample: object, coord: object, *, flip: object=True, rotate: object=False, swap: object=False, filling_value: object=160, rng: object=None) -> object:
    """augment_classifier

An explicit RNG is required; the caller retains and advances its state."""
    from sciona._runtime.dl.dsb_augmentation import augment_classifier as implementation
    if rng is None:
        raise ValueError('Explicit caller-owned RNG required')
    return implementation(sample=sample, coord=coord, flip=flip, rotate=rotate, swap=swap, filling_value=filling_value, rng=rng)

@register_atom(concrete_only)
def crop_classifier_proposal(volume: object, target: object, crop_size: object=96, stride: object=4, phase: object='test', scale_enabled: object=False, scale_limits: object=(0.85, 1.15), radius_limits: object=(6.0, 100.0), jitter_range: object=0.15, filling_value: object=160, random_state: object=None, rng: object=None) -> object:
    """Source proposal crop, jitter, scale and coordinates for the classifier.

Coordinates follow the source's clamped start and padded image dimensions;
they are not coordinates of the original unpadded volume. Python 2 integer
crop halves are preserved. Returned image is not intensity-normalized yet.

An explicit RNG is required; the caller retains and advances its state."""
    from sciona._runtime.dl.dsb_components import crop_classifier_proposal as implementation
    if rng is None:
        raise ValueError('Explicit caller-owned RNG required')
    return implementation(volume=volume, target=target, crop_size=crop_size, stride=stride, phase=phase, scale_enabled=scale_enabled, scale_limits=scale_limits, radius_limits=radius_limits, jitter_range=jitter_range, filling_value=filling_value, random_state=random_state, rng=rng)

@register_atom(concrete_only)
def assign_anchor_labels(input_size: object, target: object, boxes: object, anchors: object=(10.0, 30.0, 60.0), stride: object=4, phase: object='train', num_neg: object=800, negative_iou: object=0.02, positive_iou_train: object=0.5, positive_iou_val: object=1.0, random_state: object=None, rng: object=None) -> object:
    """Source selected-target assignment with independent surrounding boxes.

target=None denotes a negative crop. Random selection uses a local Python
RNG; deterministic seeded parity is against the source under this runtime,
not a claim about historical Python 2 random.sample sequences.

An explicit RNG is required; the caller retains and advances its state."""
    from sciona._runtime.dl.dsb_components import assign_anchor_labels as implementation
    if rng is None:
        raise ValueError('Explicit caller-owned RNG required')
    return implementation(input_size=input_size, target=target, boxes=boxes, anchors=anchors, stride=stride, phase=phase, num_neg=num_neg, negative_iou=negative_iou, positive_iou_train=positive_iou_train, positive_iou_val=positive_iou_val, random_state=random_state, rng=rng)
