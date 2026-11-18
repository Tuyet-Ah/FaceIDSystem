import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_verification import score_pairs


class VerificationMetricsTests(unittest.TestCase):
    def test_perfect_separation(self):
        labels = np.tile([True, False], 100)
        result = score_pairs(np.where(labels, .8, .1), labels)
        self.assertEqual(result['accuracyPct'], 100)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['fn'], 0)
        self.assertEqual(len(result['folds']), 10)

    def test_test_fold_does_not_select_its_own_threshold(self):
        labels = np.tile([True, False], 100)
        scores = np.where(labels, .8, .1)
        before = score_pairs(scores, labels)
        scores[:20] = np.where(labels[:20], -.8, -.9)
        after = score_pairs(scores, labels)
        self.assertEqual(before['folds'][0]['threshold'], after['folds'][0]['threshold'])
        self.assertLess(after['folds'][0]['accuracy'], 1)

    def test_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError):
            score_pairs(np.array([float('nan')]*20), np.tile([True,False], 10))


if __name__ == '__main__': unittest.main()
