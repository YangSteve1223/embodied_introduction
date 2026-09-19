"""Headless compatibility patch for state-only ManiSkill experiments.

The shared server exposes CUDA/PhysX but not NVIDIA's graphics/Vulkan stack.
State-only experiments do not need visual geometry, but SAPIEN/ManiSkill still
constructs visual builders while loading some robots and tasks. This module
skips those visual builder calls and preserves collision geometry.

Import this module before importing ``mani_skill.envs`` or constructing an
environment. It does not modify physics, rewards, success criteria, or action
controllers.
"""

from __future__ import annotations

from typing import Any


VISUAL_METHODS = (
    "add_box_visual",
    "add_capsule_visual",
    "add_cylinder_visual",
    "add_sphere_visual",
    "add_mesh_visual",
    "add_visual_from_file",
    "add_multiple_convex_visual",
)


def _skip_visual(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Preserve the builder chain while omitting visual geometry."""

    return self


def _patch_builder(builder_cls: Any) -> list[str]:
    patched = []
    for name in VISUAL_METHODS:
        if hasattr(builder_cls, name):
            setattr(builder_cls, name, _skip_visual)
            patched.append(name)
    return patched


def install_headless_visual_patch() -> dict[str, list[str]]:
    """Install the same patch used by the validated baseline script."""

    import sapien
    import sapien.render

    # Some task/robot loaders instantiate materials even when rendering is
    # disabled. The material object is unused by state-only experiments.
    null_material = lambda *args, **kwargs: None
    sapien.render.RenderMaterial = null_material

    # SAPIEN's Python modules may have imported RenderMaterial into their own
    # module namespace before ManiSkill is registered. Updating only
    # sapien.render.RenderMaterial does not update those bound references.
    material_modules = (
        "sapien.wrapper.actor_builder",
        "sapien.wrapper.urdf_loader",
        "mani_skill.utils.building.urdf_loader",
    )
    patched_material_modules = []
    for module_name in material_modules:
        try:
            module = __import__(module_name, fromlist=["RenderMaterial"])
            if hasattr(module, "RenderMaterial"):
                setattr(module, "RenderMaterial", null_material)
                patched_material_modules.append(module_name)
        except Exception as exc:  # pragma: no cover - depends on installed version
            print(f"warning: material patch {module_name}: {exc!r}")

    result: dict[str, list[str]] = {"material_modules": patched_material_modules}
    try:
        from mani_skill.utils.building.actor_builder import ActorBuilder

        result["mani_skill"] = _patch_builder(ActorBuilder)
    except Exception as exc:  # pragma: no cover - depends on installed version
        print(f"warning: ManiSkill visual patch: {exc!r}")

    try:
        from sapien.wrapper.actor_builder import ActorBuilder

        result["sapien"] = _patch_builder(ActorBuilder)
    except Exception as exc:  # pragma: no cover - depends on installed version
        print(f"warning: SAPIEN visual patch: {exc!r}")

    return result


PATCHED_BUILDERS = install_headless_visual_patch()
print(f"headless visual patch: {PATCHED_BUILDERS}")
