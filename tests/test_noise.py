import unittest

import torch

from exp002.noise import (
    InterfaceNoiseConfig,
    add_action_noise,
    add_observation_noise,
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


if __name__ == "__main__":
    unittest.main()
