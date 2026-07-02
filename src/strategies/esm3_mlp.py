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
from src.strategies.config import StrategyConfig

# Try to import ESM3 embedding extractor
try:
    from src.scripts.extract_esm3_embeddings import ESM3EmbeddingExtractor
    ESM3_AVAILABLE = True
except ImportError:
    ESM3_AVAILABLE = False


class ESM3MLPStrategy:
    """Strategy 3: Load ESM3 embeddings → Train MLP on ProMelt → Evaluate on test sets."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        run_name = config.run_name if config.run_name else "esm3_mlp"
        self.output_dir = os.path.join(config.base_output_dir, run_name)
        self.img_dir = os.path.join(config.base_output_dir, "../images", run_name)
        self.metrics: dict = {}
        self.esm3_token = None
        self.esm3_extractor = None
        self.pdb_dir = None
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

        # Check for PDB directory
        pdb_dir = Path(__file__).parent.parent.parent / "data" / "pdb_structures"
        if pdb_dir.exists() and any(pdb_dir.glob("*.pdb")):
            self.pdb_dir = str(pdb_dir)

        if self.esm3_token:
            try:
                self.esm3_extractor = ESM3EmbeddingExtractor(token=self.esm3_token)
                print(f"[ESM3-MLP] ✓ ESM3 extractor initialized")
                if self.pdb_dir:
                    print(f"[ESM3-MLP] ✓ PDB directory found: {self.pdb_dir}")
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
                pdb_dir=self.pdb_dir,
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
        print("[ESM3-MLP] Training MLP on ProMelt...")
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
        X_eval = df_eval.drop(columns=['ProteinID', 'Tm'])
        y_eval = df_eval['Tm']
        protein_ids = df_eval['ProteinID']

        y_pred = model.predict(X_eval)
        metrics = get_metrics(y_pred, y_eval)
        self.metrics[name] = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], metrics))

        print(
            f"[ESM3-MLP] {name}: RMSE {metrics[0]:.3f}, MAE {metrics[1]:.3f}, "
            f"R² {metrics[2]:.3f}, PCC {metrics[3]:.3f}, SCC {metrics[4]:.3f}"
        )

        plot_model_scatter(y_eval, y_pred, save=os.path.join(self.img_dir, f"mlp_esm3_{name.lower()}.png"))
        write_results(y_pred, y_eval, protein_ids, os.path.join(self.output_dir, f"{name.lower()}.csv"))
        save_metrics_to_json(
            dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], metrics)),
            os.path.join(self.output_dir, f"{name.lower()}.json")
        )

    def run(self) -> bool:
        """Run full ESM3-MLP pipeline."""
        print("\n" + "=" * 60)
        print("RUNNING: ESM3-MLP (Embeddings + MLP)")
        print("=" * 60)

        try:
            promelt_dir = os.path.join(self.config.data_dir, 'promelt')
            promelt_raw_dir = os.path.join(promelt_dir, 'raw')
            promelt_processed_dir = os.path.join(promelt_dir, 'processed')

            train_raw_csv = os.path.join(promelt_raw_dir, 'train_promelt_seq.csv')
            test_raw_csv = os.path.join(promelt_raw_dir, 'test_promelt_seq.csv')
            train_esm3_csv = os.path.join(promelt_processed_dir, 'dataset_esm3_train.csv')
            test_esm3_csv = os.path.join(promelt_processed_dir, 'dataset_esm3_test.csv')

            os.makedirs(promelt_processed_dir, exist_ok=True)

            # Generate split-specific ProMelt embeddings if missing
            for raw_csv, esm3_csv, split in [
                (train_raw_csv, train_esm3_csv, 'train'),
                (test_raw_csv, test_esm3_csv, 'test'),
            ]:
                if os.path.exists(raw_csv):
                    if not os.path.exists(esm3_csv):
                        print(f"[ESM3-MLP] Extracting ESM3 embeddings for ProMelt '{split}' split...")
                        self._generate_embeddings_if_missing(
                            csv_path=raw_csv,
                            output_path=esm3_csv,
                            dataset_name=f"ProMelt {split}",
                            split=split
                        )

            train = pd.read_csv(train_raw_csv)[["ProteinID", "Tm"]]
            test = pd.read_csv(test_raw_csv)[["ProteinID", "Tm"]]

            embeddings_train = self.load_embeddings_df(train_esm3_csv, "ProMelt ESM3 train").dropna()
            embeddings_test = self.load_embeddings_df(test_esm3_csv, "ProMelt ESM3 test").dropna()

            promelt_train_df = pd.merge(train, embeddings_train, on="ProteinID")
            promelt_test_df = pd.merge(test, embeddings_test, on="ProteinID")

            X_train = promelt_train_df.drop(columns=["ProteinID", "Tm"])
            y_train = promelt_train_df["Tm"]
            X_test = promelt_test_df.drop(columns=["ProteinID", "Tm"])
            y_test = promelt_test_df["Tm"]

            mlp = self.train_mlp(X_train, y_train, X_test, y_test)

            eval_dir = self.config.eval_dir
            ind_datasets = [
                ("FireProt", os.path.join(eval_dir, 'fireprot', 'raw', 'FIREPROT.csv'),
                 os.path.join(eval_dir, 'fireprot', 'processed', 'dataset_esm3.csv')),
                ("ERED_ASR", os.path.join(eval_dir, 'ered_asr', 'raw', 'ERED_ASR.csv'),
                 os.path.join(eval_dir, 'ered_asr', 'processed', 'dataset_esm3.csv')),
                ("ERED_WT", os.path.join(eval_dir, 'ered_wt', 'raw', 'ERED_WT.csv'),
                 os.path.join(eval_dir, 'ered_wt', 'processed', 'dataset_esm3.csv')),
                ("CAS", os.path.join(eval_dir, 'cas', 'raw', 'CAS.csv'),
                 os.path.join(eval_dir, 'cas', 'processed', 'dataset_esm3.csv')),
                ("HLD", os.path.join(eval_dir, 'hld', 'raw', 'HLD.csv'),
                 os.path.join(eval_dir, 'hld', 'processed', 'dataset_esm3.csv')),
                ("BRENDA", os.path.join(eval_dir, 'brenda', 'raw', 'BRENDA.csv'),
                 os.path.join(eval_dir, 'brenda', 'processed', 'dataset_esm3.csv')),
            ]

            for name, csv_path, esm_path in ind_datasets:
                try:
                    # Generate embeddings if missing
                    if not self._generate_embeddings_if_missing(
                        csv_path=csv_path,
                        output_path=esm_path,
                        dataset_name=name
                    ):
                        print(f"[ESM3-MLP] ⚠ Could not generate embeddings for {name}, skipping")
                        continue

                    labels_df = pd.read_csv(csv_path)[["ProteinID", "Tm"]]
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
