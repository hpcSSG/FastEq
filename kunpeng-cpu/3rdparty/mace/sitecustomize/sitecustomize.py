"""Install the native MACE wrapper automatically in worker processes."""
from __future__ import annotations

import os


if os.environ.get("NATIVE_CPU_MACE", "0") == "1":
    try:
        from mace.calculators import MACECalculator

        if not getattr(MACECalculator, "_native_cpu_patched", False):
            _original_init = MACECalculator.__init__

            def _native_init(self, *args, **kwargs):
                _original_init(self, *args, **kwargs)
                from fasteq.integrations.mace_runtime import install_native_mace

                install_native_mace(self)

            MACECalculator.__init__ = _native_init
            MACECalculator._native_cpu_patched = True
    except Exception as exc:
        print(f"[native-mace] sitecustomize disabled: {exc}", flush=True)
