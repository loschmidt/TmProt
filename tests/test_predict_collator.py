import unittest
from pathlib import Path

from transformers import AutoTokenizer

from src.data.tokenization import create_dataset
from src.scripts.predict_dataset import build_collator

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tmprot-1.0" / "src" / "tmprot" / "model"


class PredictCollatorTests(unittest.TestCase):
    def test_sequences_of_different_length_can_be_batched(self):
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        dataset = create_dataset(tokenizer, ["MKV", "MKVLAAGIVGLK", "MKVLA"], [1.0, 2.0, 3.0],
                                 split="test", cache_dir=None)
        batch = build_collator(tokenizer)([dataset[i] for i in range(3)])
        self.assertEqual(batch["input_ids"].shape[0], 3)
        self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
        self.assertEqual(batch["input_ids"].shape[1], len(tokenizer("MKVLAAGIVGLK")["input_ids"]))
        self.assertEqual(batch["labels"].tolist(), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
