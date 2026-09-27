"""Minimal namespace shim for RLinf's official OpenVLA loader.

The OpenVLA repository's eager top-level import pulls optional RLDS packages
that are not needed by the RLinf inference/training path.  Keeping the real
package directory on __path__ lets RLinf import only the required submodules
without changing the source repository.
"""
__path__ = [
    "/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260923/rl_pythonpath/prismatic",
    "/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/prismatic",
]
