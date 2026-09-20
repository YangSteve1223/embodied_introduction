import json
from pathlib import Path
import tempfile
import unittest

from exp003.config import load_config
from exp003.curriculum import linear_warmup_multiplier, warmup_updates


class ConfigCurriculumTests(unittest.TestCase):
    def test_registered_config(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
        config = load_config(path)
        self.assertEqual(config.experiment_id, "EXP-003")
        self.assertEqual(config.batch_size, 204_800)
        self.assertEqual(config.continuation_updates, 100)
        self.assertIsNone(config.ppo.target_kl)
        self.assertEqual(config.evaluation.seeds, (20260920, 20260921))

    def test_formal_curriculum_endpoints(self):
        self.assertEqual(warmup_updates(100, 0.2), 20)
        values = [
            linear_warmup_multiplier(index, 100, 2.0, 0.2)
            for index in range(1, 101)
        ]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[19], 2.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertTrue(all(value == 2.0 for value in values[19:]))

    def test_invalid_nondivisible_budget(self):
        registered = Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
        data = json.loads(registered.read_text(encoding="utf-8"))
        data["continuation"]["additional_timesteps"] = 123
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one batch"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
