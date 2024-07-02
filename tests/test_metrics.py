import unittest
from metrics import Metrics

class TestMetrics(unittest.TestCase):
    def test_parse_string_to_sg(self):
        metrics = Metrics()
        # Test with correct input
        pred = "(car, inFrontOf, ego)"
        sg = metrics.__parse_string_to_sg__(pred)
        target = [("car", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "(car, inFrontOf, ego), (cat, inFrontOf, ego)"
        sg = metrics.__parse_string_to_sg__(pred)
        target = [("car", "inFrontOf", "ego"), ("cat", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "[(car, inFrontOf, ego)]"
        sg = metrics.__parse_string_to_sg__(pred)
        target = [("car", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "[(car, inFrontOf, ego), (cat, inFrontOf, ego)]"
        sg = metrics.__parse_string_to_sg__(pred)
        target = [("car", "inFrontOf", "ego"), ("cat", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "[\n(car, inFrontOf, ego),\n(cat, inFrontOf, ego)\n]"
        sg = metrics.__parse_string_to_sg__(pred)
        target = [("car", "inFrontOf", "ego"), ("cat", "inFrontOf", "ego")]
        self.assertEqual(sg, target)

        # Test with incorrect input
        pred = "car, inFrontOf, ego"
        sg = metrics.__parse_string_to_sg__(pred)
        target = []
        self.assertEqual(sg, target)
        # Test with incorrect input
        pred = "(car, inFrontOf, ego"
        sg = metrics.__parse_string_to_sg__(pred)
        target = []
        self.assertEqual(sg, target)
        # Test with incorrect input
        pred = "(car, inFrontOf, ego,)"
        sg = metrics.__parse_string_to_sg__(pred)
        target = []
        self.assertEqual(sg, target)
