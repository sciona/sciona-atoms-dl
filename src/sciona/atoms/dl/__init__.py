"""Deep learning atom providers.

Atoms derived from DL competition solutions. Framework-agnostic atoms
use numpy/scipy only. Framework-aware atoms require optional torch dependency.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
