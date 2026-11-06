"""Explicit Web Traffic lifecycle and reusable numerical boundaries."""
import numpy as np
from sciona.ghost.registry import register_atom


def concrete_only(*args,**kwargs):
    raise ValueError('Concrete runtime inputs required; symbolic execution is not certified')


@register_atom(concrete_only)
def unpack(payload: dict) -> tuple[object,object,float,int,int,list,list,list,list,np.ndarray]:
    """Explicit unpack operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_contract import unpack
    return unpack(payload)


@register_atom(concrete_only)
def prepare(frame: object,valid_threshold: float,add_days: int,corr_backoffset: int) -> tuple[object,object,np.ndarray,np.ndarray]:
    """Explicit prepare operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_features import prepare
    return prepare(frame,valid_threshold,add_days,corr_backoffset)


@register_atom(concrete_only)
def feature_axes(logged: object) -> tuple[np.ndarray,list,object,object]:
    """Explicit feature axes operation for qualified runtime composition."""
    return logged.to_numpy(copy=True),logged.index.tolist(),logged.columns[0],logged.columns[-1]


@register_atom(concrete_only)
def correlations(values: np.ndarray,starts: np.ndarray,ends: np.ndarray,lags: list,thresholds: list,backoffset: int) -> np.ndarray:
    """Explicit correlations operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_features import correlations
    return correlations(values,starts,ends,lags,thresholds,backoffset)


@register_atom(concrete_only)
def popularity(logged: object) -> object:
    """Explicit popularity operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_features import popularity
    return popularity(logged)


@register_atom(concrete_only)
def page_metadata(row_labels: list) -> tuple[np.ndarray,dict]:
    """Explicit page metadata operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_features import page_metadata
    return page_metadata(row_labels)


@register_atom(concrete_only)
def calendar(begin: object,end: object,add_days: int) -> tuple[np.ndarray,np.ndarray,object]:
    """Explicit calendar operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_features import calendar
    return calendar(begin,end,add_days)


@register_atom(concrete_only)
def assemble(logged: object,missing: object,correlation_columns: np.ndarray,page_map: np.ndarray,encoded: dict,popularity_values: object,dow: np.ndarray,lagged_ix: np.ndarray,features_end: object) -> tuple[dict,dict]:
    """Explicit assemble operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_features import assemble
    return assemble(logged,missing,correlation_columns,page_map,encoded,popularity_values,dow,lagged_ix,features_end)


@register_atom(concrete_only)
def configure(tensors: dict,plain: dict,config: object) -> object:
    """Explicit configure operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_lifecycle import configure
    return configure(tensors,plain,config)


@register_atom(concrete_only)
def initialize(prepared: object) -> object:
    """Explicit initialize operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_lifecycle import initialize
    return initialize(prepared)


@register_atom(concrete_only)
def advance(state: dict,batches: list,observations: list) -> tuple[dict,list]:
    """Explicit advance operation for qualified runtime composition."""
    from sciona.webtraffic_lifecycle import advance
    return advance(state,batches,observations)


@register_atom(concrete_only)
def save_checkpoint(checkpoints: list,state: dict) -> list:
    """Explicit save checkpoint operation for qualified runtime composition."""
    from sciona.webtraffic_lifecycle import save_checkpoint
    return save_checkpoint(checkpoints,state)


@register_atom(concrete_only)
def train(prepared: object,initialized: object,advance_function: object,checkpoint_function: object) -> dict:
    """Explicit train operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_lifecycle import train
    return train(prepared,initialized,advance_function,checkpoint_function)


@register_atom(concrete_only)
def prediction_batches(prepared: object) -> list:
    """Explicit prediction batches operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_lifecycle import prediction_batches
    return prediction_batches(prepared)


@register_atom(concrete_only)
def checkpoint_forecast(trained: dict,records: list,row_labels: list) -> object:
    """Explicit checkpoint forecast operation for qualified runtime composition."""
    import pandas as pd
    from sciona.webtraffic_lifecycle import predict_checkpoints
    return predict_checkpoints(trained["checkpoints"],records,pd.Index(row_labels))[0]


@register_atom(concrete_only)
def date_forecast(prepared: object,frame: object) -> object:
    """Explicit date forecast operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_lifecycle import date_forecast
    return date_forecast(prepared,frame)


@register_atom(concrete_only)
def forecast_axes(frame: object) -> tuple[np.ndarray,list,list]:
    """Explicit forecast axes operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_contract import forecast_axes
    return forecast_axes(frame)


@register_atom(concrete_only)
def aligned_output(values: np.ndarray,row_labels: list,column_labels: list,output_rows: list,output_columns: list,identifiers: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Explicit aligned output operation for qualified runtime composition."""
    from sciona.panel_output import aligned_suffix_records
    return aligned_suffix_records(values,row_labels,column_labels,output_rows,output_columns,identifiers)


@register_atom(concrete_only)
def result(identifiers: np.ndarray,values: np.ndarray) -> dict:
    """Explicit result operation for qualified runtime composition."""
    from sciona.decomposed_webtraffic_contract import result
    return result(identifiers,values)
