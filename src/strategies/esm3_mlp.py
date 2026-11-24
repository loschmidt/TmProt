import os
import sys
import ast
import json
import traceback
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from dotenv import load_dotenv

from src.eval.metrics import get_metrics
from src.eval.visualization import plot_model_scatter
from src.eval.io import save_metrics_to_json, write_results
from src.eval.eval_sets import STRATEGY_SETS, raw_csv_path
from src.strategies.config import StrategyConfig

# Try to import ESM3 embedding extractor
try:
    from src.scripts.extract_esm3_embeddings import ESM3EmbeddingExtractor
    ESM3_AVAILABLE = True
except ImportError:
    ESM3_AVAILABLE = False


class ESM3MLPStrategy:
    """Strategy 3: Load ESM3 embeddings → Train MLP on Meltome → Evaluate on test sets."""

    def __init__(self, config: StrategyConfig, esm3_model: str = "esm3-large-2024-03"):
        self.config = config
        self.esm3_model = esm3_model
        run_name = config.run_name if config.run_name else "esm3_mlp"
        self.output_dir = os.path.join(config.base_output_dir, run_name)
        self.img_dir = os.path.join(config.base_output_dir, "../images", run_name)
        self.metrics: dict = {}
        self.esm3_token = None
        self.esm3_extractor = None
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        # Initialize ESM3 if available
        self._init_esm3()

    def load_embeddings_df(self, esm_path: str, name: str = "Dataset") -> pd.DataFrame:
        """Load embeddings from parquet or CSV."""
        is_parquet = esm_path.endswith('.parquet')

        if not os.path.exists(esm_path):
            fallback_path = (
                esm_path.replace('.parquet', '.csv')
                if is_parquet
                else esm_path.replace('.csv', '.parquet')
            )
            if os.path.exists(fallback_path):
                esm_path = fallback_path
                is_parquet = fallback_path.endswith('.parquet')
            else:
                raise FileNotFoundError(f"Neither {esm_path} nor {fallback_path} found")

        if is_parquet:
            embeddings_df = pd.read_parquet(esm_path, engine='pyarrow')
            if 'embedding' in embeddings_df.columns:
                embeddings_df['embedding'] = embeddings_df['embedding'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                embedding_df = pd.DataFrame(embeddings_df['embedding'].tolist())
                embeddings_df = pd.concat(
                    [embeddings_df[['ProteinID']].reset_index(drop=True), embedding_df], axis=1
                )
                print(f"Loaded {name} embeddings from parquet with {embedding_df.shape[1]} dimensions")
            else:
                print(f"Loaded {name} embeddings from parquet (no 'embedding' column to expand)")
        else:
            embeddings_df = pd.read_csv(esm_path)
            if 'embedding' in embeddings_df.columns:
                embeddings_df['embedding'] = embeddings_df['embedding'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
                embedding_df = pd.DataFrame(embeddings_df['embedding'].tolist())
                embeddings_df = pd.concat(
                    [embeddings_df[['ProteinID']].reset_index(drop=True), embedding_df], axis=1
                )
                print(f"Loaded {name} embeddings from csv with {embedding_df.shape[1]} dimensions")
            else:
                print(f"Loaded {name} embeddings from csv")

        return embeddings_df

    def _init_esm3(self):
        """Initialize ESM3 embedding extractor if available."""
        if not ESM3_AVAILABLE:
            return

        load_dotenv()
        self.esm3_token = os.getenv("ESM3_API_TOKEN")

        if self.esm3_token:
            try:
                self.esm3_extractor = ESM3EmbeddingExtractor(
                    model_name=self.esm3_model, token=self.esm3_token
                )
                print(f"[ESM3-MLP] ✓ ESM3 extractor initialized ({self.esm3_model})")
            except Exception as e:
                print(f"[ESM3-MLP] ⚠ Could not initialize ESM3 extractor: {e}")
                self.esm3_extractor = None

    def _generate_embeddings_if_missing(self, csv_path: str, output_path: str, dataset_name: str, split: str = None) -> bool:
        """
        Check if embeddings exist. If not and ESM3 is available, generate them.

        Args:
            csv_path: Path to input CSV with sequences
            output_path: Path to save embeddings
            dataset_name: Name for logging
            split: Optional split (train/val/test) for filtered extraction

        Returns:
            True if embeddings exist or were generated, False otherwise
        """
        # Check if embeddings already exist
        if os.path.exists(output_path):
            print(f"[ESM3-MLP] ✓ {dataset_name} embeddings already exist")
            return True

        # Try to generate if ESM3 is available
        if self.esm3_extractor is None:
            print(f"[ESM3-MLP] ⚠ {dataset_name} embeddings not found and ESM3 not available")
            print(f"[ESM3-MLP] To generate embeddings:")
            print(f"[ESM3-MLP]   1. Get token: https://forge.evolutionaryscale.ai/apikeys")
            print(f"[ESM3-MLP]   2. Add to .env: ESM3_API_TOKEN=your_token")
            print(f"[ESM3-MLP]   3. Run: python src/scripts/extract_esm3_embeddings.py \\")
            print(f"[ESM3-MLP]        -f {csv_path} \\")
            print(f"[ESM3-MLP]        -o {output_path}" + (f" \\\n[ESM3-MLP]        --split {split}" if split else ""))
            return False

        print(f"[ESM3-MLP] → Generating {dataset_name} embeddings via ESM3 (split={split})...")
        try:
            result_df = self.esm3_extractor.extract_from_csv(
                csv_path=csv_path,
                output_path=output_path,
                split=split,
            )
            if len(result_df) > 0:
                print(f"[ESM3-MLP] ✓ Generated embeddings for {len(result_df)} proteins")
                return True
            else:
                print(f"[ESM3-MLP] ✗ Failed to generate embeddings")
                return False
        except Exception as e:
            print(f"[ESM3-MLP] ✗ Error generating embeddings: {e}")
            return False

    def train_mlp(self, X_train, y_train, X_test, y_test) -> MLPRegressor:
        """Train MLP on training set."""
        print("[ESM3-MLP] Training MLP on Meltome...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            random_state=self.config.random_state
        )
        mlp.fit(X_train, y_train)

        train_metrics = get_metrics(mlp.predict(X_train), y_train)
        test_metrics = get_metrics(mlp.predict(X_test), y_test)

        print(f"[ESM3-MLP] Train RMSE: {train_metrics[0]:.3f}, MAE: {train_metrics[1]:.3f}, R²: {train_metrics[2]:.3f}, PCC: {train_metrics[3]:.3f}, SCC: {train_metrics[4]:.3f}")
        print(f"[ESM3-MLP] Test  RMSE: {test_metrics[0]:.3f}, MAE: {test_metrics[1]:.3f}, R²: {test_metrics[2]:.3f}, PCC: {test_metrics[3]:.3f}, SCC: {test_metrics[4]:.3f}")

        save_metrics_to_json(
            dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], train_metrics)),
            os.path.join(self.output_dir, "train.json")
        )
        save_metrics_to_json(
            dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], test_metrics)),
            os.path.join(self.output_dir, "test.json")
        )
        joblib.dump(mlp, os.path.join(self.output_dir, "mlp.joblib"))

        return mlp

    def evaluate(self, model: MLPRegressor, name: str, df_eval: pd.DataFrame) -> None:
        """Evaluate model on a named dataset."""
        print(f"[ESM3-MLP] Evaluating on {name}")
        X_eval = df_eval.drop(columns=[c for c in ('ProteinID', 'Tm', 'Length') if c in df_eval.columns])
        y_eval = df_eval['Tm']
        protein_ids = df_eval['ProteinID']
        length = df_eval['Length'] if 'Length' in df_eval.columns else None

        y_pred = model.predict(X_eval)
        metrics = get_metrics(y_pred, y_eval)
        self.metrics[name] = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], metrics))

        print(
            f"[ESM3-MLP] {name}: RMSE {metrics[0]:.3f}, MAE {metrics[1]:.3f}, "
            f"R² {metrics[2]:.3f}, PCC {metrics[3]:.3f}, SCC {metrics[4]:.3f}"
        )

        plot_model_scatter(y_eval, y_pred, save=os.path.join(self.img_dir, f"mlp_esm3_{name.lower()}.png"))
        write_results(y_pred, y_eval, protein_ids, os.path.join(self.output_dir, f"{name.lower()}.csv"),
                      length=length, seed=self.config.random_state)
        save_metrics_to_json(
            dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], metrics)),
            os.path.join(self.output_dir, f"{name.lower()}.json")
        )

    def save_split_predictions(self, model: MLPRegressor, name: str, raw_csv: str, esm3_csv: str) -> None:
        """Write per-protein predictions for a Meltome split (no metrics, no pooling).

        Proteins without an ESM3 embedding are dropped by the merge, so the
        saved file can have fewer rows than the split manifest.
        """
        labels = pd.read_csv(raw_csv)[["ProteinID", "Tm", "Length"]]
        df = pd.merge(labels, self.load_embeddings_df(esm3_csv, f"Meltome ESM3 {name}").dropna(), on="ProteinID")
        X = df.drop(columns=["ProteinID", "Tm", "Length"])
        write_results(model.predict(X), df["Tm"], df["ProteinID"],
                      os.path.join(self.output_dir, f"{name}.csv"),
                      length=df["Length"], seed=self.config.random_state)

    def run(self) -> bool:
        """Run full ESM3-MLP pipeline."""
        print("\n" + "=" * 60)
        print("RUNNING: ESM3-MLP (Embeddings + MLP)")
        print("=" * 60)

        try:
            meltome_processed_dir = os.path.join(self.config.data_dir, 'meltome', 'processed')

            # Melters-only, no length cap (the project-wide "drop the cap"
            # direction) -- ESM3 runs sequence-only, which has no length cap
            # at all, so it does not need to truncate like the ESM2 strategies.
            # Embedding filenames are disambiguated by model size so large
            # and medium runs can coexist without overwriting each other.
            model_tag = self.esm3_model.split("-")[1]  # "esm3-large-2024-03" -> "large"
            train_raw_csv = os.path.join(meltome_processed_dir, 'train_meltome.csv')
            test_raw_csv = os.path.join(meltome_processed_dir, 'test_meltome.csv')
            train_esm3_csv = os.path.join(meltome_processed_dir, f'dataset_esm3_{model_tag}_train.csv')
            test_esm3_csv = os.path.join(meltome_processed_dir, f'dataset_esm3_{model_tag}_test.csv')

            os.makedirs(meltome_processed_dir, exist_ok=True)

            # Generate split-specific Meltome embeddings if missing
            for raw_csv, esm3_csv, split in [
                (train_raw_csv, train_esm3_csv, 'train'),
                (test_raw_csv, test_esm3_csv, 'test'),
            ]:
                if os.path.exists(raw_csv) and not os.path.exists(esm3_csv):
                    print(f"[ESM3-MLP] Extracting ESM3 embeddings for Meltome '{split}' split...")
                    self._generate_embeddings_if_missing(
                        csv_path=raw_csv,
                        output_path=esm3_csv,
                        dataset_name=f"Meltome {split}",
                        split=split
                    )

            train = pd.read_csv(train_raw_csv)[["ProteinID", "Tm"]]
            test = pd.read_csv(test_raw_csv)[["ProteinID", "Tm"]]

            embeddings_train = self.load_embeddings_df(train_esm3_csv, "Meltome ESM3 train").dropna()
            embeddings_test = self.load_embeddings_df(test_esm3_csv, "Meltome ESM3 test").dropna()

            # TMPROT_TRAIN_VAL=1 folds the val split into training (final-model
            # retrain with hyperparameters already fixed). Test is untouched.
            if os.environ.get("TMPROT_TRAIN_VAL") == "1":
                val = pd.read_csv(os.path.join(meltome_processed_dir, 'val_meltome.csv'))[["ProteinID", "Tm"]]
                embeddings_val = self.load_embeddings_df(
                    os.path.join(meltome_processed_dir, f'dataset_esm3_{model_tag}_val.csv'),
                    "Meltome ESM3 val",
                ).dropna()
                train = pd.concat([train, val], ignore_index=True)
                embeddings_train = pd.concat([embeddings_train, embeddings_val], ignore_index=True)
                print(f"[ESM3-MLP] TMPROT_TRAIN_VAL=1: training on train+val ({len(train)} labels)")

            meltome_train_df = pd.merge(train, embeddings_train, on="ProteinID")
            meltome_test_df = pd.merge(test, embeddings_test, on="ProteinID")

            X_train = meltome_train_df.drop(columns=["ProteinID", "Tm"])
            y_train = meltome_train_df["Tm"]
            X_test = meltome_test_df.drop(columns=["ProteinID", "Tm"])
            y_test = meltome_test_df["Tm"]

            mlp = self.train_mlp(X_train, y_train, X_test, y_test)

            # Val is saved only when it was held out; with train+val it is in-sample.
            self.save_split_predictions(mlp, "test", test_raw_csv, test_esm3_csv)
            if os.environ.get("TMPROT_TRAIN_VAL") != "1":
                self.save_split_predictions(
                    mlp, "val",
                    os.path.join(meltome_processed_dir, 'val_meltome.csv'),
                    os.path.join(meltome_processed_dir, f'dataset_esm3_{model_tag}_val.csv'),
                )

            eval_dir = self.config.eval_dir

            # Buckets (see src/eval/eval_sets.py): esm3 evaluates protherm +
            # independent + long_esm2 (CAS). Sequence-only ESM3 has no length
            # cap (verified working past 2000 AA against the live API), so
            # unlike the earlier structure-informed design, CAS's long
            # sequences (1049-1535 AA) are no longer out of reach -- esm3_mlp
            # scores the same sets as baseline_mlp/esm2_lora.
            for name in STRATEGY_SETS["esm3_mlp"]:
                csv_path = raw_csv_path(eval_dir, name)
                esm_path = os.path.join(eval_dir, name, 'processed', f'dataset_esm3_{model_tag}.csv')
                try:
                    # No length filtering: sequence-only ESM3 has no cap, so
                    # every row -- including CAS's 1049-1535 AA proteins --
                    # scores directly against the raw CSV.
                    labels_df = pd.read_csv(csv_path)[["ProteinID", "Tm", "Length"]]
                    # Generate embeddings if missing
                    if not self._generate_embeddings_if_missing(
                        csv_path=csv_path,
                        output_path=esm_path,
                        dataset_name=name,
                    ):
                        print(f"[ESM3-MLP] ⚠ Could not generate embeddings for {name}, skipping")
                        continue
                    embeddings_df = self.load_embeddings_df(esm_path, name)
                    df_eval = pd.merge(labels_df, embeddings_df, on="ProteinID")
                    self.evaluate(mlp, name, df_eval)
                except FileNotFoundError as e:
                    print(f"[ESM3-MLP] ⚠ {name} not found: {e}")

            print("[ESM3-MLP] ✓ Strategy completed successfully!")
            return True

        except Exception as e:
            print(f"[ESM3-MLP] ✗ Error: {e}")
            traceback.print_exc()
            return False
