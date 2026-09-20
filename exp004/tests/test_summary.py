import unittest

from exp004.config import ARMS, TRAINING_SEEDS, load_config
from exp004.summarize import load_stage, summarize_endpoint
from pathlib import Path
import tempfile
import json


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("exp004/configs/experiment.json")

    def make_cells(self):
        cells = {}
        for seed in TRAINING_SEEDS:
            for model in (*ARMS, "C4"):
                for condition in ("clean", "combined4x"):
                    for eval_seed in self.config.evaluation.seeds:
                        success = 0.50
                        if model == "C1" and condition == "combined4x":
                            success = 0.60
                        if model == "C1" and condition == "clean":
                            success = 0.49
                        cells[(seed, model, 100, condition, eval_seed)] = {
                            "metrics_mean": {"success_once": success, "success_at_end": success, "return": 50.0, "episode_len": 100.0},
                            "interface_metrics": {"noise_induced_clip_fraction": 0.01},
                        }
        return cells

    def test_gate_is_matched_to_c0(self):
        summary = summarize_endpoint(self.make_cells(), self.config, "primary", 100, (*ARMS, "C4"))
        c1 = summary["aggregate"]["C1"]
        self.assertAlmostEqual(c1["robustness_gain"]["mean"], 0.10)
        self.assertAlmostEqual(c1["clean_cost"]["mean"], -0.01)
        self.assertTrue(summary["engineering_success_gate"]["C1"]["all_gates_pass"])

    def test_stage_loader_rejects_wrong_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "secondary" / "seed1001" / "model-C0" / "update-149" / "eval-clean" / "eval_seed20260920.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"experiment_id": "EXP-004", "stage": "secondary", "smoke": False, "model": "C0", "checkpoint_update": 149, "training_seed": 1001, "evaluation_condition": "clean", "evaluation_multiplier": 0.0, "evaluation_seed": 20260920, "episodes_collected": 256, "metrics_mean": {"success_once": 0.1, "success_at_end": 0.1, "return": 1.0, "episode_len": 100.0}, "interface_metrics": {"noise_induced_clip_fraction": 0.0}}))
            with self.assertRaises(ValueError):
                load_stage(root, "secondary", self.config)


if __name__ == "__main__":
    unittest.main()
