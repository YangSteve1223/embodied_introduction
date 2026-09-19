import unittest

import torch

from exp002.noise import (
    InterfaceNoiseConfig,
    PEG_INSERTION_STATE_LAYOUT,
    add_action_noise,
    add_observation_noise,
    add_peg_insertion_observation_noise,
)


class NoiseTests(unittest.TestCase):
    def test_disabled_noise_is_deterministic(self):
        config = InterfaceNoiseConfig(enabled=False)
        obs = torch.ones(4, 10)
        action = torch.zeros(4, 7)
        self.assertTrue(torch.equal(add_observation_noise(obs, config), obs))
        self.assertTrue(torch.equal(add_action_noise(action, config), action))

    def test_action_noise_is_bounded_and_clipped(self):
        config = InterfaceNoiseConfig(
            enabled=True,
            action_sigma=10.0,
            action_clip=0.05,
        )
        action = torch.zeros(128, 7)
        result = add_action_noise(action, config)
        self.assertLessEqual(float(result.max()), 0.05)
        self.assertGreaterEqual(float(result.min()), -0.05)

    def test_observation_scale_broadcasts(self):
        config = InterfaceNoiseConfig(
            enabled=True,
            obs_sigma=1.0,
            obs_clip=0.1,
        )
        obs = torch.zeros(8, 3)
        scale = torch.tensor([1.0, 2.0, 3.0])
        result = add_observation_noise(obs, config, scale=scale)
        self.assertEqual(tuple(result.shape), (8, 3))
        self.assertTrue(torch.all(result.abs() <= 0.3))

    def test_peg_insertion_layout_and_geometry_are_preserved(self):
        config = InterfaceNoiseConfig(enabled=True)
        obs = torch.zeros(4, PEG_INSERTION_STATE_LAYOUT.total_dim)
        obs[:, PEG_INSERTION_STATE_LAYOUT.peg_half_size] = 0.04
        obs[:, PEG_INSERTION_STATE_LAYOUT.box_hole_radius] = 0.02
        result = add_peg_insertion_observation_noise(obs, config)
        self.assertEqual(tuple(result.shape), (4, 43))
        self.assertTrue(
            torch.equal(
                result[:, PEG_INSERTION_STATE_LAYOUT.peg_half_size],
                obs[:, PEG_INSERTION_STATE_LAYOUT.peg_half_size],
            )
        )
        self.assertTrue(
            torch.equal(
                result[:, PEG_INSERTION_STATE_LAYOUT.box_hole_radius],
                obs[:, PEG_INSERTION_STATE_LAYOUT.box_hole_radius],
            )
        )

    def test_peg_insertion_quaternions_remain_normalized(self):
        config = InterfaceNoiseConfig(enabled=True)
        obs = torch.zeros(8, PEG_INSERTION_STATE_LAYOUT.total_dim)
        for pose in (
            PEG_INSERTION_STATE_LAYOUT.tcp_pose,
            PEG_INSERTION_STATE_LAYOUT.peg_pose,
            PEG_INSERTION_STATE_LAYOUT.box_hole_pose,
        ):
            obs[:, pose.start + 3] = 1.0
        result = add_peg_insertion_observation_noise(obs, config)
        for pose in (
            PEG_INSERTION_STATE_LAYOUT.tcp_pose,
            PEG_INSERTION_STATE_LAYOUT.peg_pose,
            PEG_INSERTION_STATE_LAYOUT.box_hole_pose,
        ):
            quaternion = result[:, pose.start + 3 : pose.stop]
            self.assertTrue(
                torch.allclose(
                    torch.linalg.vector_norm(quaternion, dim=-1),
                    torch.ones(8),
                    atol=1e-6,
                )
            )

    def test_peg_insertion_noise_rejects_wrong_state_width(self):
        config = InterfaceNoiseConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, "expected 43 state features"):
            add_peg_insertion_observation_noise(torch.zeros(2, 42), config)


if __name__ == "__main__":
    unittest.main()
