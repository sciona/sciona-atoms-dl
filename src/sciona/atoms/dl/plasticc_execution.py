"""Community Avocado execution boundaries; runtime inputs supplied by caller."""
from sciona.ghost.registry import register_atom


def witness_plasticc_prepare(payload: dict) -> dict:
    return {'kind': 'Plasticc.Prepared'}


@register_atom(witness_plasticc_prepare)
def plasticc_prepare(payload: dict) -> object:
    """Validate versioned training, prediction, empirical reference and config."""
    from sciona.plasticc_runtime import prepare_runtime
    return prepare_runtime(payload)


def witness_plasticc_train(prepared: object) -> dict:
    if prepared != {'kind': 'Plasticc.Prepared'}: raise ValueError('Prepared runtime required')
    return {'kind': 'Plasticc.Completed'}


@register_atom(witness_plasticc_train)
def plasticc_train(prepared: object) -> object:
    """Augment observations, fit GPs, extract features and train grouped folds."""
    from sciona.plasticc_runtime import train_runtime
    return train_runtime(prepared)


def witness_plasticc_predict(completed: object) -> dict:
    if completed != {'kind': 'Plasticc.Completed'}: raise ValueError('Completed runtime required')
    return {'kind': 'Plasticc.Predictions'}


@register_atom(witness_plasticc_predict)
def plasticc_predict(completed: object) -> dict:
    """Featurize separate prediction objects and produce adjusted probabilities."""
    from sciona.plasticc_runtime import predict_runtime
    return predict_runtime(completed)
