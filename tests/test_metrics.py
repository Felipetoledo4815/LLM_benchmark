import unittest
from utils.metrics import Metrics

class TestMetrics(unittest.TestCase):

    def test_recall(self):
        m = Metrics()

        # Perfect recall
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 1.0)

        # Undefined recall
        pred = [("car", "inFrontOf", "ego")]
        target = []
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 0.0)

        # Partial recall
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego"), ("car", "within_25m", "ego")]
        recall = m.calculate_recall(pred, target)
        self.assertEqual(recall, 0.5)

    def test_precision(self):
        m = Metrics()

        # Perfect precision
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 1.0)

        # Undefined precision
        pred = []
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 0.0)

        # Partial precision
        pred = [("car", "inFrontOf", "ego"), ("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        precision = m.calculate_precision(pred, target)
        self.assertEqual(precision, 0.5)


    def test_f1(self):
        m = Metrics()

        # Perfect f1
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego")]
        f1 = m.calculate_f1(pred, target)
        self.assertEqual(f1, 1.0)

        # Undefined f1
        pred = []
        target = [("car", "inFrontOf", "ego")]
        f1 = m.calculate_f1(pred, target)
        self.assertEqual(f1, 0.0)

        # Partial f1
        pred = [("car", "inFrontOf", "ego")]
        target = [("car", "inFrontOf", "ego"), ("car", "within_25m", "ego")]
        f1 = m.calculate_f1(pred, target)
        self.assertAlmostEqual(f1, 0.6666, places=3)


    def test_heatmaps(self):
        #TODO: Implement test_heatmaps
        pass
