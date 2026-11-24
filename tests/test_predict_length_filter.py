import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.data.loading import load_filtered_dataset
from src.scripts.predict_dataset import parse_args


class PredictLengthFilterTests(unittest.TestCase):
    def parse(self, *extra):
        with mock.patch.object(sys, "argv", ["predict_dataset.py", "--inputs", "a.csv", "--outdir", "out", *extra]):
            return parse_args()[0]

    def test_filter_is_off_by_default(self):
        self.assertFalse(self.parse().length_filter)

    def test_filter_can_be_switched_on(self):
        self.assertTrue(self.parse("--length_filter").length_filter)

    def test_old_flag_is_still_accepted(self):
        self.assertFalse(self.parse("--no_length_filter").length_filter)

    def test_all_proteins_are_kept_without_filter_and_length_column(self):
        df = pd.DataFrame({"ProteinID": ["a", "b", "c"], "sequence": ["A" * 5, "A" * 500, "A" * 2500], "Tm": [40.0, 50.0, 60.0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.csv")
            df.to_csv(path, index=False)
            self.assertEqual(len(load_filtered_dataset(path, False)), 3)


if __name__ == "__main__":
    unittest.main()
