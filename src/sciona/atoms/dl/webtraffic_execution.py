"""Community Web Traffic full runtime boundaries; runtime data supplied by caller."""
from sciona.ghost.registry import register_atom


def witness_webtraffic_prepare(payload: dict) -> dict:
    return {'kind':'WebTraffic.PreparedRuntime'}


@register_atom(witness_webtraffic_prepare)
def webtraffic_prepare(payload: dict) -> object:
    """Validate runtime series/config and prepare complete source-derived features."""
    from sciona.webtraffic_payload import decode_payload
    return decode_payload(payload)


def witness_webtraffic_train(prepared: object) -> dict:
    if prepared!={'kind':'WebTraffic.PreparedRuntime'}:raise ValueError('Prepared runtime required')
    return {'kind':'WebTraffic.TrainedRuntime'}


@register_atom(witness_webtraffic_train)
def webtraffic_train(prepared: object) -> object:
    """Train three GRU models, update declared EMA snapshots and retain checkpoints."""
    from sciona.webtraffic_payload import train_payload
    return train_payload(prepared)


def witness_webtraffic_forecast(completed: object) -> dict:
    if completed!={'kind':'WebTraffic.TrainedRuntime'}:raise ValueError('Trained runtime required')
    return {'kind':'WebTraffic.ForecastPayload'}


@register_atom(witness_webtraffic_forecast)
def webtraffic_forecast(completed: object) -> dict:
    """Forecast with saved EMA checkpoints and encode dated integer ensemble output."""
    from sciona.webtraffic_payload import encode_forecast
    return encode_forecast(completed)
