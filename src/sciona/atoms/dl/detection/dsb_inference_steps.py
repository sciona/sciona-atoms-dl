# DSB recipe arithmetic adapted from lfz/DSB2017.
# MIT License
#
# Copyright (c) 2023 lfz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Draft explicit DSB inference recipe steps; not catalog publication approval."""
import numpy as np
import torch
from sciona.ghost.registry import register_atom


def concrete_only(*args, **kwargs):
    raise ValueError('Concrete model state and geometry required for this DSB recipe step')


@register_atom(concrete_only)
def prepare_inference_geometry(processed_volume: object, tile_side: int=144, margin: int=32) -> tuple:
    """Pad to detector stride and form global coordinates before separate tiling.

    Source constants are explicit outputs. Intensity normalization occurs later.
    Coordinate axes span the full padded volume, never each tile independently.
    """
    from sciona.dsb_components import _integer
    side=_integer(tile_side,'tile_side');border=_integer(margin,'margin',0)
    volume=np.asarray(processed_volume)
    if (volume.ndim!=4 or min(volume.shape)<1 or volume.dtype.kind not in 'fiu'
            or not np.isfinite(volume).all() or side%16 or border%16 or side<=border):
        raise ValueError('Finite CZYX volume and aligned detector geometry required')
    shape=tuple(((n+3)//4)*4 for n in volume.shape[1:])
    padded=np.pad(volume,[(0,0)]+[(0,p-n) for p,n in zip(shape,volume.shape[1:])],
                  mode='constant',constant_values=170)
    coordinates=np.stack(np.meshgrid(*(np.linspace(-.5,.5,n//4) for n in shape),indexing='ij')).astype(np.float32)
    return padded,coordinates,side,16,border,side//4,4,border//4,4


@register_atom(concrete_only)
def normalize_and_align_tiles(tiles: object, image_grid: object, coordinate_tiles: object, coordinate_grid: object) -> tuple:
    """Check paired tile grids and apply source float32 intensity scaling."""
    if tuple(image_grid)!=tuple(coordinate_grid):raise ValueError('Image and coordinate grids differ')
    return (np.asarray(tiles).astype(np.float32)-128)/128,coordinate_tiles,image_grid


@register_atom(concrete_only)
def predict_detector_tiles(tiles: object, coordinates: object, detector: object, tile_batch_size: int=1) -> object:
    """Run an explicit eval detector in ordered batches without changing its state."""
    from sciona.dsb_components import _integer
    batch=_integer(tile_batch_size,'tile_batch_size')
    if detector.training:raise ValueError('Eval detector required')
    if len(tiles)!=len(coordinates) or not len(tiles):raise ValueError('Aligned nonempty tile arrays required')
    parameter=next(detector.parameters());outputs=[]
    with torch.inference_mode():
        for start in range(0,len(tiles),batch):
            x=torch.from_numpy(tiles[start:start+batch]).to(parameter)
            c=torch.from_numpy(coordinates[start:start+batch]).to(parameter)
            outputs.append(detector(x,c).cpu().numpy())
    return np.concatenate(outputs)


@register_atom(concrete_only)
def decode_detector_proposals(output: object) -> object:
    """Decode the source anchors at stride four and strict score threshold -3."""
    from sciona.dsb_components import decode_proposals
    proposals,_=decode_proposals(output,-3.)
    return proposals


@register_atom(concrete_only)
def filter_classifier_proposals(proposals: object) -> object:
    """Retain source proposals with score strictly greater than -1 before NMS."""
    values=np.asarray(proposals)
    if values.ndim!=2 or values.shape[1]!=5 or not np.isfinite(values).all():
        raise ValueError('Finite score and cubic-box rows required')
    return values[values[:,0]>-1.]


@register_atom(concrete_only)
def suppress_classifier_proposals(proposals: object) -> object:
    """Apply source cubic-box NMS at overlap threshold .05."""
    from sciona.dsb_components import suppress_proposals
    return suppress_proposals(proposals,.05)


@register_atom(concrete_only)
def prepare_classifier_crops(processed_volume: object, proposals: object, topk: int=5, crop_size: int=96) -> tuple:
    """Form source top-k crops and coordinate channels, retaining empty slots."""
    from sciona.dsb_components import prepare_classifier_batch
    crops,coordinates,_=prepare_classifier_batch(processed_volume,proposals,topk,crop_size)
    return crops,coordinates


@register_atom(concrete_only)
def predict_classifier_crops(crops: object, coordinates: object, classifier: object) -> tuple:
    """Run the explicit eval classifier; preserve case and per-proposal scores."""
    if classifier.training:raise ValueError('Eval classifier required')
    parameter=next(classifier.parameters())
    with torch.inference_mode():
        _,case,each=classifier(torch.from_numpy(crops[None]).to(parameter),
                               torch.from_numpy(coordinates[None]).to(parameter))
    return case.cpu().numpy(),each.cpu().numpy()
