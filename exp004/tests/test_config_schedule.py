import json
from pathlib import Path
import tempfile
import unittest

from exp004.config import load_config, primary_matrix, secondary_matrix, training_matrix
from exp004.curriculum import linear_warmup_multiplier


class ConfigScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
        cls.config = load_config(cls.path)

    def test_registered_budget_and_endpoints(self):
        self.assertEqual(self.config.batch_size, 204800)
        self.assertEqual(self.config.continuation.total_updates, 150)
        self.assertEqual(self.config.continuation.additional_timesteps, 30720000)
        self.assertEqual(self.config.continuation.primary_update, 100)
        self.assertEqual(self.config.continuation.secondary_update, 150)

    def test_fixed_twenty_update_warmup(self):
        values = [linear_warmup_multiplier(i, 3.0, 20) for i in range(1, 151)]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[19], 3.0)
        self.assertEqual(values[20], 3.0)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_matrix_counts_and_unique_keys(self):
        training = training_matrix(self.config)
        self.assertEqual(len(training), 12)
        self.assertEqual(len({(x["training_seed"], x["arm"]) for x in training}), 12)
        primary = primary_matrix(self.config)
        secondary = secondary_matrix(self.config)
        self.assertEqual(len(primary), 60)
        self.assertEqual(len(secondary), 48)
        self.assertEqual(len({tuple(sorted(x.items())) for x in primary}), 60)
        self.assertTrue(all(x["model"] == "C4" and x["checkpoint_update"] == 100 or x["model"] != "C4" for x in primary))

    def test_invalid_warmup_or_budget_rejected(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["continuation"]["warmup_updates"] = 30
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
