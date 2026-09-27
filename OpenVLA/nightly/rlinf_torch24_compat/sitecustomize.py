"""Small user-space compatibility shim for RLinf on the server's PyTorch 2.4.

RLinf v0.4 imports the DTensor symbols from the public
``torch.distributed.tensor`` namespace, while this server's PyTorch 2.4 build
still exposes the implementation under ``torch.distributed._tensor``.
Nothing here changes the installed packages; it only aliases the compatible
symbols for this experiment process.
"""

import os
import sys


# Keep the server's ManiSkill environment as the owner of torch (2.4.1), but
# make the already-installed OpenVLA dependencies (for example draccus)
# visible later on sys.path.  Putting that environment first would silently
# replace torch with its older 2.2 build.
_openvla_site = "/share/yangpengju-local/anaconda3/envs/openvla-oft-repro/lib/python3.10/site-packages"
if os.path.isdir(_openvla_site) and _openvla_site not in sys.path:
    sys.path.append(_openvla_site)


try:
    import torch.distributed.tensor as public_tensor
    import torch.distributed._tensor as legacy_tensor
    import torch.distributed._tensor.placement_types as legacy_placements

    for name in (
        "DTensor",
        "DeviceMesh",
        "Placement",
        "Replicate",
        "Shard",
        "Partial",
    ):
        if not hasattr(public_tensor, name) and hasattr(legacy_tensor, name):
            setattr(public_tensor, name, getattr(legacy_tensor, name))

    sys.modules.setdefault(
        "torch.distributed.tensor.placement_types", legacy_placements
    )
except Exception as exc:  # pragma: no cover - startup must remain non-fatal
    pass


# The server has CUDA compute but no usable Vulkan render device.  This is
# only for state-observation ManiSkill runs; it preserves collision geometry
# and skips optional visual mesh/material construction.
try:
    if os.environ.get("RLINF_HEADLESS_MANISKILL") != "1":
        raise RuntimeError("RLINF_HEADLESS_MANISKILL is not enabled")
    import sapien
    import sapien.render

    sapien.render.RenderMaterial = lambda *args, **kwargs: None

    def _skip_visual(self, *args, **kwargs):
        return self

    import mani_skill.envs
    import mani_skill.render.utils as _render_utils

    _render_utils.can_render = lambda *args, **kwargs: False

    _visual_methods = (
        "add_box_visual",
        "add_capsule_visual",
        "add_cylinder_visual",
        "add_sphere_visual",
        "add_mesh_visual",
        "add_visual_from_file",
        "add_multiple_convex_visual",
    )

    def _skip_visual(self, *args, **kwargs):
        return self

    from mani_skill.utils.building.actor_builder import ActorBuilder

    for _name in _visual_methods:
        if hasattr(ActorBuilder, _name):
            setattr(ActorBuilder, _name, _skip_visual)
    try:
        from sapien.wrapper.actor_builder import ActorBuilder as SapienActorBuilder

        for _name in _visual_methods:
            if hasattr(SapienActorBuilder, _name):
                setattr(SapienActorBuilder, _name, _skip_visual)
    except Exception:
        pass
except Exception as exc:  # pragma: no cover - optional simulator patch
    pass
