"""Explicit grapheme lifecycle stages; catalog qualification remains pending."""
from sciona.ghost.registry import register_atom


def witness_inputs(validation_images: object, query_images: object) -> tuple:
    return ({'kind': 'GrayscaleImages'}, {'kind': 'GrayscaleImages'})


@register_atom(witness_inputs)
def inference_inputs(validation_images: object, query_images: object) -> tuple:
    return validation_images, query_images


def witness_ood(ood_spec: object) -> dict:
    return {'kind': 'B7State', 'branch': 'ood'}


@register_atom(witness_ood)
def train_ood(ood_spec: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import train_b7
    if ood_spec.get('branch') != 'ood':
        raise ValueError('OOD training specification required')
    return train_b7(**ood_spec)


def witness_seen(seen_spec: object) -> dict:
    return {'kind': 'B7State', 'branch': 'seen'}


@register_atom(witness_seen)
def train_seen(seen_spec: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import train_b7
    if seen_spec.get('branch') != 'seen':
        raise ValueError('Seen training specification required')
    return train_b7(**seen_spec)


def witness_font_a(font_a_spec: object) -> dict:
    return {'kind': 'FontState'}


@register_atom(witness_font_a)
def train_font_a(font_a_spec: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import train_font
    return train_font(**font_a_spec)


def witness_font_b(font_b_spec: object) -> dict:
    return {'kind': 'FontState'}


@register_atom(witness_font_b)
def train_font_b(font_b_spec: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import train_font
    return train_font(**font_b_spec)


def witness_generator_a(gan_a_spec: object, classifier_state: object) -> dict:
    if classifier_state != {'kind': 'FontState'}:
        raise ValueError('Font checkpoint state required')
    return {'kind': 'GeneratorState'}


@register_atom(witness_generator_a)
def train_generator_a(gan_a_spec: object, classifier_state: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import train_generator
    return train_generator(classifier_state=classifier_state, **gan_a_spec)


def witness_generator_b(gan_b_spec: object, classifier_state: object) -> dict:
    return witness_generator_a(gan_b_spec, classifier_state)


@register_atom(witness_generator_b)
def train_generator_b(gan_b_spec: object, classifier_state: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import train_generator
    return train_generator(classifier_state=classifier_state, **gan_b_spec)


def witness_b7_prediction(state: object, images: object) -> dict:
    if state not in ({'kind': 'B7State', 'branch': 'ood'}, {'kind': 'B7State', 'branch': 'seen'}):
        raise ValueError('B7 checkpoint state required')
    return {'kind': 'Logits', 'classes': 1295 if state['branch'] == 'ood' else 14784}


@register_atom(witness_b7_prediction)
def predict_b7(state: object, images: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import predict_b7 as predict
    return predict(state, images)


def witness_generated_prediction(state: object, images: object) -> dict:
    if state != {'kind': 'GeneratorState'}:
        raise ValueError('Generator checkpoint state required')
    return {'kind': 'Logits', 'classes': 14784}


@register_atom(witness_generated_prediction)
def predict_generated(state: object, images: object) -> object:
    from sciona._runtime.dl.grapheme_workflow_runtime import predict_generated as predict
    return predict(state, images)


def witness_scores(ood_logits: object, seen_logits: object) -> dict:
    if ood_logits != {'kind': 'Logits', 'classes': 1295} or seen_logits != {'kind': 'Logits', 'classes': 14784}:
        raise ValueError('Aligned OOD and seen logit contracts required')
    return {'kind': 'ClassifierScores'}


@register_atom(witness_scores)
def classifier_scores(ood_logits: object, seen_logits: object) -> object:
    from sciona._runtime.dl.grapheme_classifier_inference import classifier_predictions
    return classifier_predictions(ood_logits, seen_logits)


def witness_ensemble(logits_a: object, logits_b: object) -> dict:
    if logits_a != {'kind': 'Logits', 'classes': 14784} or logits_b != logits_a:
        raise ValueError('Two full joint-class logit contracts required')
    return {'kind': 'UnseenPredictions'}


@register_atom(witness_ensemble)
def font_ensemble(logits_a: object, logits_b: object) -> object:
    from sciona._runtime.dl.bengali_unseen_ensemble import marginal_predictions
    return marginal_predictions(logits_a, logits_b)


def witness_calibration(classifier_scores: object, unseen_predictions: object, truth_components: object,
                        known_class_mask: object, sample_weight: object, thresholds: object) -> dict:
    if classifier_scores != {'kind': 'ClassifierScores'} or unseen_predictions != {'kind': 'UnseenPredictions'}:
        raise ValueError('Complete validation branch predictions required')
    return {'kind': 'RoutingCalibration'}


@register_atom(witness_calibration)
def fit_router(classifier_scores: object, unseen_predictions: object, truth_components: object,
               known_class_mask: object, sample_weight: object, thresholds: object) -> object:
    from sciona._runtime.dl.grapheme_routing_calibration import calibrate_grapheme_router
    return calibrate_grapheme_router(classifier_scores['ood_confidence'], classifier_scores['seen_joint_classes'],
        unseen_predictions['joint_classes'].numpy(), truth_components, seen_class_mask=known_class_mask,
        sample_weight=sample_weight, thresholds=thresholds)


def witness_routing(classifier_scores: object, unseen_predictions: object, calibration: object,
                    known_class_mask: object) -> dict:
    if (classifier_scores != {'kind': 'ClassifierScores'} or unseen_predictions != {'kind': 'UnseenPredictions'}
            or calibration != {'kind': 'RoutingCalibration'}):
        raise ValueError('Complete query predictions and fitted calibration required')
    return {'kind': 'JointPredictions'}


@register_atom(witness_routing)
def route_predictions(classifier_scores: object, unseen_predictions: object, calibration: object,
                      known_class_mask: object) -> object:
    from sciona._runtime.dl.bengali_routing import route_predictions as route
    return route(classifier_scores['ood_confidence'], classifier_scores['seen_joint_classes'],
                 unseen_predictions['joint_classes'].numpy(), seen_class_mask=known_class_mask,
                 threshold=calibration['threshold'])


def witness_submission(joint_classes: object) -> dict:
    if joint_classes != {'kind': 'JointPredictions'}:
        raise ValueError('Routed joint predictions required')
    return {'kind': 'CanonicalComponents'}


@register_atom(witness_submission)
def submission_components(joint_classes: object) -> object:
    from sciona._runtime.dl.grapheme_routing_calibration import submission_components as convert
    return convert(joint_classes)
