import unittest

try:
    import torch
except ImportError:  # Local Mac intentionally has no project PyTorch environment.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is validated in the server maniskill environment")
class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exp003.config import NoiseBaseConfig

        cls.base = NoiseBaseConfig(
            qpos_sigma=0.003,
            qpos_clip=0.01,
            qvel_sigma=0.01,
            qvel_clip=0.03,
            position_sigma=0.002,
            position_clip=0.006,
            orientation_sigma_deg=0.5,
            orientation_clip_deg=1.5,
            action_sigma=0.02,
            action_clip=0.05,
        )

    def test_zero_multiplier_is_exact(self):
        from exp003.noise import (
            apply_action_treatment,
            apply_observation_treatment,
            build_treatment,
        )

        treatment = build_treatment(self.base, "combined", 0.0)
        obs = torch.zeros(2, 43)
        action = torch.tensor([[1.2, 0, 0, 0, 0, 0, 0.0]])
        policy_obs = apply_observation_treatment(obs, treatment)
        trace = apply_action_treatment(
            action,
            treatment,
            torch.full((7,), -1.0),
            torch.full((7,), 1.0),
        )
        self.assertTrue(torch.equal(policy_obs, obs))
        self.assertTrue(torch.equal(trace.noise_delta, torch.zeros_like(action)))
        self.assertEqual(float(trace.applied[0, 0]), 1.0)

    def test_component_masks_and_scaling(self):
        from exp003.noise import build_treatment

        observation = build_treatment(self.base, "observation", 2.0)
        action = build_treatment(self.base, "action", 4.0)
        combined = build_treatment(self.base, "combined", 2.0)
        self.assertEqual(observation.config.qpos_sigma, 0.006)
        self.assertEqual(observation.config.action_sigma, 0.0)
        self.assertEqual(action.config.qpos_sigma, 0.0)
        self.assertEqual(action.config.action_sigma, 0.08)
        self.assertEqual(combined.config.position_sigma, 0.004)
        self.assertEqual(combined.config.action_clip, 0.1)

    def test_observation_geometry_is_preserved(self):
        from exp002.noise import PEG_INSERTION_STATE_LAYOUT
        from exp003.noise import apply_observation_treatment, build_treatment

        treatment = build_treatment(self.base, "observation", 1.0)
        obs = torch.zeros(8, 43)
        obs[:, PEG_INSERTION_STATE_LAYOUT.peg_half_size] = 0.04
        obs[:, PEG_INSERTION_STATE_LAYOUT.box_hole_radius] = 0.02
        for pose in (
            PEG_INSERTION_STATE_LAYOUT.tcp_pose,
            PEG_INSERTION_STATE_LAYOUT.peg_pose,
            PEG_INSERTION_STATE_LAYOUT.box_hole_pose,
        ):
            obs[:, pose.start + 3] = 1.0
        noisy = apply_observation_treatment(
            obs,
            treatment,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(
            torch.equal(
                noisy[:, PEG_INSERTION_STATE_LAYOUT.peg_half_size],
                obs[:, PEG_INSERTION_STATE_LAYOUT.peg_half_size],
            )
        )
        self.assertTrue(
            torch.equal(
                noisy[:, PEG_INSERTION_STATE_LAYOUT.box_hole_radius],
                obs[:, PEG_INSERTION_STATE_LAYOUT.box_hole_radius],
            )
        )


if __name__ == "__main__":
    unittest.main()
