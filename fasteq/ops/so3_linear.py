"""Compatibility shim for the historical ``fasteq.ops.so3_linear`` import path.

The implementation now lives in :mod:`fasteq.ops.eqlinear` and dispatches to the CUTLASS
backends.  Existing call sites keep importing ``fast_equi_linear`` from here.
"""

from ..triton.so3_linear import (  # noqa: F401  (historical module-level API)
    LinearLayout,
    so3_linear_auto,
    so3_linear_backward_x,
)
from .eqlinear import (  # noqa: F401
    eqlinear_clear_plan_cache,
    eqlinear_set_plan_cache,
    fast_equi_linear,
)

__all__ = [
    "fast_equi_linear",
    "eqlinear_clear_plan_cache",
    "eqlinear_set_plan_cache",
    "LinearLayout",
    "so3_linear_auto",
    "so3_linear_backward_x",
]
