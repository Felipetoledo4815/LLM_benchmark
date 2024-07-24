import unittest
from utils.utils import parse_string_to_sg

class TestUtils(unittest.TestCase):

    def test_parse_string_to_sg(self):

        # Test with correct input
        pred = "(car, inFrontOf, ego)"
        sg = parse_string_to_sg(pred)
        target = [("car", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "(car, inFrontOf, ego), (cat, inFrontOf, ego)"
        sg = parse_string_to_sg(pred)
        target = [("car", "inFrontOf", "ego"), ("cat", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "[(car, inFrontOf, ego)]"
        sg = parse_string_to_sg(pred)
        target = [("car", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "[(car, inFrontOf, ego), (cat, inFrontOf, ego)]"
        sg = parse_string_to_sg(pred)
        target = [("car", "inFrontOf", "ego"), ("cat", "inFrontOf", "ego")]
        self.assertEqual(sg, target)
        # Test with correct input
        pred = "[\n(car, inFrontOf, ego),\n(cat, inFrontOf, ego)\n]"
        sg = parse_string_to_sg(pred)
        target = [("car", "inFrontOf", "ego"), ("cat", "inFrontOf", "ego")]
        self.assertEqual(sg, target)

        # Test partially incorrect input
        pred = "[\n(car, inFrontOf, ego),\n(cat, inFron"
        sg = parse_string_to_sg(pred)
        target = [("car", "inFrontOf", "ego")]
        self.assertEqual(sg, target)

        # Test with incorrect input
        pred = "car, inFrontOf, ego"
        sg = parse_string_to_sg(pred)
        target = []
        self.assertEqual(sg, target)
        # Test with incorrect input
        pred = "(car, inFrontOf, ego"
        sg = parse_string_to_sg(pred)
        target = []
        self.assertEqual(sg, target)
        # Test with incorrect input
        pred = "(car, inFrontOf, ego,)"
        sg = parse_string_to_sg(pred)
        target = []
        self.assertEqual(sg, target)
        # Test with incorrect input
        pred = "(car, inFro"
        sg = parse_string_to_sg(pred)
        target = []
        self.assertEqual(sg, target)
