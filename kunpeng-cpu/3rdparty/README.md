# Third-party integration

This directory contains the optional MACE integration:

```text
mace/
  integrations/     # Packaged as fasteq.integrations
  sitecustomize/    # Optional startup hook for existing MACE applications
```

These files are integration code from the user's CPU implementation, not a vendored MACE source checkout. Install MACE separately with the version required by your model. No empty cuEquivariance or SevenNet submodules are declared.

For an existing CPU MACECalculator, import `install_native_mace` from `fasteq.integrations.mace_runtime`. The packaging configuration maps that import to `mace/integrations`.
