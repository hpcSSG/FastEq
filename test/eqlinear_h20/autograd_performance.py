"""Public forward with a pre-existing differentiable input for every candidate."""
from __future__ import annotations

from performance_bindings import CurrentTorchFamily


class PublicAutogradFamily:
    FILES = ("autograd_performance.py",)

    @staticmethod
    def expected_calls(family):
        return [key for key in CurrentTorchFamily.expected_calls(family)
                if key.split("/")[1:] == ["forward_autograd", "public_warm"]]

    def __init__(self, case, session, family="current"):
        self.base = CurrentTorchFamily(case, session, family)
        self.calls = [call for call in self.base.calls if call.key in self.expected_calls(family)]
        if sorted(call.key for call in self.calls) != sorted(self.expected_calls(family)):
            raise ValueError("Public autograd call scope differs")
        self.setup_metadata = {"input_leaf": "x.requires_grad=True leaf prepared outside every measured call for all candidates",
                               "weight": "constant canonical weight without gradient",
                               "measurement": "new differentiable forward graph and output allocation per call; no backward in this window"}

    def close(self):
        self.base.close()
