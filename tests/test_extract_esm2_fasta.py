import os
import tempfile
import unittest

import pandas as pd
from Bio import SeqIO

from src.scripts.extract_esm2_embeddings import csv_to_fasta


class CsvToFastaLengthTests(unittest.TestCase):
    def test_no_length_filter_keeps_short_and_long_sequences(self):
        rows = {"short": "A" * 5, "edge_low": "A" * 20, "mid": "A" * 500, "long": "A" * 2500}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "in.csv")
            fasta_path = os.path.join(tmp, "out.fasta")
            pd.DataFrame({"ProteinID": list(rows), "sequence": list(rows.values())}).to_csv(csv_path, index=False)
            self.assertTrue(csv_to_fasta(csv_path, fasta_path))
            kept = {r.id: len(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
        self.assertEqual(kept, {k: len(v) for k, v in rows.items()})

    def test_missing_and_empty_sequences_are_still_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "in.csv")
            fasta_path = os.path.join(tmp, "out.fasta")
            pd.DataFrame({"ProteinID": ["ok", "nan", "blank"], "sequence": ["MKV", None, "   "]}).to_csv(csv_path, index=False)
            self.assertTrue(csv_to_fasta(csv_path, fasta_path))
            kept = [r.id for r in SeqIO.parse(fasta_path, "fasta")]
        self.assertEqual(kept, ["ok"])


if __name__ == "__main__":
    unittest.main()
