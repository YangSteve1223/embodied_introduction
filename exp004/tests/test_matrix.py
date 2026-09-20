import unittest

from exp004.config import cross_matrix, load_config, primary_matrix, secondary_matrix


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("exp004/configs/experiment.json")

    def test_primary_has_c4_reference_only(self):
        rows = primary_matrix(self.config)
        c4 = [row for row in rows if row["model"] == "C4"]
        self.assertEqual(len(c4), 12)
        self.assertTrue(all(row["checkpoint_update"] == 100 for row in c4))
        self.assertEqual({row["multiplier"] for row in c4}, {0.0, 4.0})

    def test_secondary_excludes_c4(self):
        rows = secondary_matrix(self.config)
        self.assertFalse(any(row["model"] == "C4" for row in rows))
        self.assertTrue(all(row["checkpoint_update"] == 150 for row in rows))

    def test_cross_grid_is_optional_and_complete(self):
        rows = cross_matrix(self.config)
        self.assertEqual(len(rows), 162)
        self.assertEqual({row["condition"] for row in rows}, {"combined1x", "combined2x", "combined3x"})
        self.assertTrue(all(row["model"] == "C4" and row["checkpoint_update"] == 100 or row["model"] != "C4" for row in rows))

    def test_primary_clean_and_ood_pair(self):
        rows = primary_matrix(self.config)
        for seed in self.config.training_seeds:
            for model in (*self.config.arms, "C4"):
                selected = [row for row in rows if row["training_seed"] == seed and row["model"] == model]
                self.assertEqual({row["condition"] for row in selected}, {"clean", "combined4x"})
                self.assertEqual({row["evaluation_seed"] for row in selected}, set(self.config.evaluation.seeds))


if __name__ == "__main__":
    unittest.main()
