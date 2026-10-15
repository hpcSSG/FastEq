"""Kunpeng CPU edition. Native compilation is deferred until first use."""
import os

__version__ = "0.2.0+kunpeng.1"
_BACKEND = os.environ.get("FASTEQ_BACKEND", "cpu").strip().lower()
if _BACKEND != "cpu":
    raise RuntimeError("This FastEq edition supports FASTEQ_BACKEND=cpu only.")
__all__ = ["__version__", "_BACKEND"]
