"""
Shared evaluation-set registry for the three Tm strategies (TmProt 2.0).

Single source of truth for which external benchmarks each strategy evaluates and
how the comparison is kept fair. Three buckets by design:

  1. "protherm"    — ProTherm, homolog-removed (PROTHERM_clean.csv).
  2. "independent" — FireProt, ERED_WT, ERED_ASR, HLD, BRENDA.
  3. "long_esm2"   — CAS. Kept as its own bucket because ESM2-based strategies
                     truncate at 1022 residues and 27/29 CAS proteins are longer,
                     so instances differ from buckets 1-2 in what the model sees.

No evaluation set is length-filtered. ESM2 strategies truncate at 1022 residues (the only
length effect); sequence-only ESM3 sees the full sequence.

Putting metrics from different buckets in one number would compare different
instance sets across models, so it is avoided; compare_models.py reports per
bucket instead.
"""
import os

# Each entry: display name -> (raw/processed CSV path, subdirectory name)
# dirname is used to locate processed embeddings under eval_dir/<dirname>.
# The path is expressed relative to the eval root and combined at runtime so
# strategies using different roots stay consistent.
EVAL_SETS = {
    # name    dirname       filename (or None to use dirname.upper())
    "protherm":  {"dirname": "protherm",  "file": "PROTHERM_clean"},
    "fireprot":  {"dirname": "fireprot",  "file": "FIREPROT"},
    "ered_wt":   {"dirname": "ered_wt",   "file": "ERED_WT"},
    "ered_asr":  {"dirname": "ered_asr",  "file": "ERED_ASR"},
    "hld":       {"dirname": "hld",       "file": "HLD"},
    "brenda":    {"dirname": "brenda",    "file": "BRENDA"},
    "cas":       {"dirname": "cas",       "file": "CAS"},
}

# Bucket 1: every strategy scores the full set.
PROTHERM_BUCKET = ["protherm"]

# Bucket 2: every strategy scores these in full.
INDEPENDENT_BUCKET = ["fireprot", "ered_wt", "ered_asr", "hld", "brenda"]

# Bucket 3: ESM2-based strategies only (CAS is structure-length incompatible).
LONG_ESM2_BUCKET = ["cas"]

# Which sets each strategy evaluates (by name).
STRATEGY_SETS = {
    "baseline_mlp": PROTHERM_BUCKET + INDEPENDENT_BUCKET + LONG_ESM2_BUCKET,
    "esm2_lora":    PROTHERM_BUCKET + INDEPENDENT_BUCKET + LONG_ESM2_BUCKET,
    "esm3_mlp":     PROTHERM_BUCKET + INDEPENDENT_BUCKET + LONG_ESM2_BUCKET,
}


def bucket_of(name: str) -> str:
    """Return the bucket ("protherm" | "independent" | "long_esm2") for a set name."""
    if name in PROTHERM_BUCKET:
        return "protherm"
    if name in INDEPENDENT_BUCKET:
        return "independent"
    if name in LONG_ESM2_BUCKET:
        return "long_esm2"
    raise ValueError(f"Unknown eval set: {name}")


def raw_csv_path(eval_root: str, name: str) -> str:
    """Absolute-ish path to the source CSV for a set under an eval root."""
    meta = EVAL_SETS[name]
    # ProTherm is held in <root>/protherm/processed/; others in <root>/<name>/raw/.
    if meta["dirname"] == "protherm":
        rel = f"protherm/processed/{meta['file']}.csv"
    else:
        rel = f"{meta['dirname']}/raw/{meta['file']}.csv"
    return os.path.join(eval_root, rel)
