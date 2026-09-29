"""Cartesian spherical harmonics retain e3nn's polynomial code and autograd.

The upstream implementation is generated for l <= 12 and contains a large
degree-specialized polynomial. Keeping it avoids truncating the supported
degrees while the smaller angle-based harmonics have a dedicated Triton path.
"""

from e3nn.o3._spherical_harmonics import (
    SphericalHarmonics, spherical_harmonics, _spherical_harmonics,
)


__all__ = ["SphericalHarmonics", "spherical_harmonics", "_spherical_harmonics"]
