"""Compatibility aliases for the server's PyTorch 2.4 distributed tensor API."""
try:
    import sys
    import torch.distributed.tensor as _public_tensor
    from torch.distributed._tensor import DTensor, DeviceMesh, distribute_tensor
    import torch.distributed._tensor.placement_types as _placement_types

    _public_tensor.DTensor = DTensor
    _public_tensor.DeviceMesh = DeviceMesh
    _public_tensor.distribute_tensor = distribute_tensor
    sys.modules.setdefault("torch.distributed.tensor.placement_types", _placement_types)
except Exception:
    pass

# The official OpenVLA class exposes `_supports_sdpa` as a property that reads
# `language_model._supports_sdpa`.  Transformers 4.56 probes that property
# during the parent constructor, before OpenVLA has created `language_model`.
# The model path used here does not need SDPA, so force the eager attention
# implementation before the parent constructor performs its probe.
try:
    from rlinf.models.embodiment.openvla_oft.official.openvla_oft_action_model import (
        OpenVLAOFTForRLActionPrediction,
    )

    _openvla_original_init = OpenVLAOFTForRLActionPrediction.__init__

    def _openvla_eager_init(self, config):
        config._attn_implementation_internal = "eager"
        return _openvla_original_init(self, config)

    OpenVLAOFTForRLActionPrediction.__init__ = _openvla_eager_init
except Exception:
    pass
