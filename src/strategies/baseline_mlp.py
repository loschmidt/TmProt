import os
import subprocess
import traceback
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor

import mlflow
import mlflow.sklearn
import dagshub
from mlflow.models import infer_signature

from src.eval.metrics import get_metrics
from src.eval.visualization import plot_model_scatter
from src.eval.io import save_metrics_to_json, write_results
from src.strategies.config import StrategyConfig


class BaselineMLPStrategy:
    """Strategy 1: Extract ESM2 → Train MLP on ProMelt → Evaluate on test sets."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        run_name = config.run_name if config.run_name else "baseline_mlp"
        self.output_dir = os.path.join(config.base_output_dir, run_name)
        self.img_dir = os.path.join(config.base_output_dir, "../images", run_name)
        self.metrics: dict = {}
        self._pooled_pred: list = []
        self._pooled_true: list = []
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

    def extract_embeddings(self, input_csv: str, output_path: str, split: str = None) -> bool:
        """Extract ESM2 embeddings using extract_esm2_embeddings.py"""
        script_path = Path(__file__).parent.parent / "scripts" / "extract_esm2_embeddings.py"
        project_root = Path(__file__).parent.parent.parent

        input_csv_abs = str(project_root / input_csv) if not os.path.isabs(input_csv) else input_csv
        output_path_abs = str(project_root / output_path) if not os.path.isabs(output_path) else output_path
        output_dir_abs = os.path.dirname(output_path_abs)

        cmd = [
            "python", str(script_path),
            "-f", input_csv_abs,
            "-o", output_dir_abs,
            "--model_name", "esm2_t33_650M_UR50D",
            "--repr_layer", "33",
        ]
        if split:
            cmd.extend(["--split", split])

        split_suffix = f"_{split}" if split else ""
        output_file = f"dataset_esm2{split_suffix}.csv"
        print(f"[Baseline MLP] Extracting ESM2 embeddings from {input_csv} (split={split})...")
        print(f"[Baseline MLP] Output: {output_path} → {output_file}")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        
        try:
            subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
            if os.path.exists(output_path_abs):
                print(f"[Baseline MLP] ✓ Embeddings saved to {output_path_abs}")
                return True
            else:
                print(f"[Baseline MLP] ✗ Embeddings not found at {output_path_abs}")
                return False
        except subprocess.CalledProcessError as e:
            print(f"[Baseline MLP] ✗ Embedding extraction failed: {e}")
            return False

    def load_embeddings_df(self, esm_path: str) -> pd.DataFrame:
        """Load embeddings CSV."""
        if not os.path.exists(esm_path):
            raise FileNotFoundError(f"Embeddings not found at {esm_path}")
        return pd.read_csv(esm_path)

    def train_mlp(self, X_train, y_train, X_test, y_test) -> MLPRegressor:
        """Train MLP on training set."""
        print("[Baseline MLP] Training MLP on ProMelt...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            random_state=self.config.random_state
        )
        mlp.fit(X_train, y_train)

        train_metrics = get_metrics(mlp.predict(X_train), y_train)
        test_metrics = get_metrics(mlp.predict(X_test), y_test)

        print(f"[Baseline MLP] Train RMSE: {train_metrics[0]:.3f}, MAE: {train_metrics[1]:.3f}, R²: {train_metrics[2]:.3f}, PCC: {train_metrics[3]:.3f}, SCC: {train_metrics[4]:.3f}")
        print(f"[Baseline MLP] Test  RMSE: {test_metrics[0]:.3f}, MAE: {test_metrics[1]:.3f}, R²: {test_metrics[2]:.3f}, PCC: {test_metrics[3]:.3f}, SCC: {test_metrics[4]:.3f}")

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
        X_eval = df_eval.drop(columns=['ProteinID', 'Tm'])
        y_eval = df_eval['Tm']
        protein_ids = df_eval['ProteinID']

        y_pred = model.predict(X_eval)
        metrics = get_metrics(y_pred, y_eval)
        self.metrics[name] = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], metrics))

        self._pooled_pred.extend(y_pred.tolist())
        self._pooled_true.extend(y_eval.tolist())

        plot_model_scatter(y_eval, y_pred, save=os.path.join(self.img_dir, f"mlp_esm2_{name.lower()}.png"))
        write_results(y_pred, y_eval, protein_ids, os.path.join(self.output_dir, f"{name.lower()}.csv"))
        save_metrics_to_json(
            dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], metrics)),
            os.path.join(self.output_dir, f"{name.lower()}.json")
        )

    def _log_to_mlflow(
        self,
        run_name: str,
        mlp: MLPRegressor,
        X_train,
        promelt_train_df: pd.DataFrame,
        promelt_test_df: pd.DataFrame,
    ) -> None:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        dagshub.init(
            repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
            repo_name=os.environ["DAGSHUB_REPO_NAME"],
            mlflow=True,
        )
        mlflow.set_experiment("baseline_mlp")

        mlflow.start_run(run_name=run_name)
        try:
            mlflow.log_params({
                "hidden_layer_sizes": "64,32",
                "early_stopping": True,
                "random_state": self.config.random_state,
                "esm2_model": "esm2_t33_650M_UR50D",
                "repr_layer": 33,
                "n_train": len(promelt_train_df),
                "n_test": len(promelt_test_df),
            })

            train_dataset = mlflow.data.from_pandas(promelt_train_df, name="promelt_train", targets="Tm")
            test_dataset = mlflow.data.from_pandas(promelt_test_df, name="promelt_test", targets="Tm")
            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(test_dataset, context="validation")

            for dataset, m in self.metrics.items():
                prefix = dataset.lower().replace("-", "_")
                for k, v in m.items():
                    mlflow.log_metric(f"{prefix}/{k}", v)

            signature = infer_signature(X_train, mlp.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=mlp,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:3],
                registered_model_name="baseline_mlp",
            )

            env_path = os.path.join(self.output_dir, "pip_freeze.txt")
            try:
                pip_output = subprocess.check_output(["pip", "freeze"], text=True)
            except Exception:
                pip_output = subprocess.check_output(["pip3", "freeze"], text=True)
            with open(env_path, "w") as f:
                f.write(pip_output)
            mlflow.log_artifact(env_path, artifact_path="environment")

            pyproject = str(Path(__file__).parent.parent.parent / "pyproject.toml")
            if os.path.exists(pyproject):
                mlflow.log_artifact(pyproject, artifact_path="environment")

            for dataset_name in [k for k in self.metrics if k != "Combined"]:
                dirname = dataset_name.lower().replace("-", "_")
                img_path = os.path.join(self.img_dir, f"mlp_esm2_{dirname}.png")
                if os.path.exists(img_path):
                    mlflow.log_artifact(img_path, artifact_path="plots")
                csv_path = os.path.join(self.output_dir, f"{dirname}.csv")
                if os.path.exists(csv_path):
                    mlflow.log_artifact(csv_path, artifact_path="predictions")

        finally:
            mlflow.end_run(status="FINISHED")

    def run(self) -> bool:
        """Run full Baseline MLP pipeline."""
        print("\n" + "=" * 60)
        print("RUNNING: Baseline MLP (ESM2 + MLP)")
        print("=" * 60)

        try:
            promelt_dir = os.path.join(self.config.data_dir, 'promelt')
            promelt_raw_dir = os.path.join(promelt_dir, 'raw')
            promelt_processed_dir = os.path.join(promelt_dir, 'processed')

            train_raw_csv = os.path.join(promelt_raw_dir, 'train_promelt_seq.csv')
            test_raw_csv = os.path.join(promelt_raw_dir, 'test_promelt_seq.csv')
            train_esm2_csv = os.path.join(promelt_processed_dir, 'dataset_esm2_train.csv')
            test_esm2_csv = os.path.join(promelt_processed_dir, 'dataset_esm2_test.csv')

            os.makedirs(promelt_processed_dir, exist_ok=True)

            for raw_csv, split_csv, split in [
                (train_raw_csv, train_esm2_csv, 'train'),
                (test_raw_csv, test_esm2_csv, 'test'),
            ]:
                if not os.path.exists(split_csv):
                    print(f"[Baseline MLP] Extracting ESM2 embeddings for ProMelt '{split}' split...")
                    if not self.extract_embeddings(raw_csv, split_csv, split):
                        return False

            train = pd.read_csv(train_raw_csv)[["ProteinID", "Tm"]]
            test = pd.read_csv(test_raw_csv)[["ProteinID", "Tm"]]

            embeddings_train = self.load_embeddings_df(train_esm2_csv).dropna()
            embeddings_test = self.load_embeddings_df(test_esm2_csv).dropna()

            promelt_train_df = pd.merge(train, embeddings_train, on="ProteinID")
            promelt_test_df = pd.merge(test, embeddings_test, on="ProteinID")

            X_train = promelt_train_df.drop(columns=["ProteinID", "Tm"])
            y_train = promelt_train_df["Tm"]
            X_test = promelt_test_df.drop(columns=["ProteinID", "Tm"])
            y_test = promelt_test_df["Tm"]

            mlp = self.train_mlp(X_train, y_train, X_test, y_test)

            eval_datasets = {
                "FireProt": "fireprot",
                "ERED_ASR": "ered_asr",
                "ERED_WT":  "ered_wt",
                "CAS":      "cas",
                "HLD":      "hld",
                "BRENDA":   "brenda",
            }
            eval_dir = self.config.eval_dir

            for name, dirname in eval_datasets.items():
                csv_path = os.path.join(eval_dir, dirname, 'raw', f'{name.upper()}.csv')
                esm_path = os.path.join(eval_dir, dirname, 'processed', 'dataset_esm2.csv')
                try:
                    if not os.path.exists(esm_path):
                        print(f"[Baseline MLP] Extracting ESM2 embeddings for {name}...")
                        if not self.extract_embeddings(csv_path, esm_path):
                            print(f"[Baseline MLP] ⚠ {name}: embedding extraction failed, skipping.")
                            continue
                    labels_df = pd.read_csv(csv_path)[["ProteinID", "Tm"]]
                    embeddings_df = self.load_embeddings_df(esm_path)
                    df_eval = pd.merge(labels_df, embeddings_df, on="ProteinID")
                    self.evaluate(mlp, name, df_eval)
                except FileNotFoundError as e:
                    print(f"[Baseline MLP] ⚠ {name} not found: {e}")

            if self._pooled_pred:
                combined = get_metrics(self._pooled_pred, self._pooled_true)
                self.metrics["Combined"] = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], combined))
                save_metrics_to_json(self.metrics["Combined"], os.path.join(self.output_dir, "combined.json"))

            if self.config.use_mlflow:
                try:
                    self._log_to_mlflow(
                        run_name=os.path.basename(self.output_dir),
                        mlp=mlp,
                        X_train=X_train,
                        promelt_train_df=promelt_train_df,
                        promelt_test_df=promelt_test_df,
                    )
                except Exception as e:
                    print(f"[Baseline MLP] ⚠ MLflow logging failed: {e}")

            print("[Baseline MLP] ✓ Strategy completed successfully!")
            return True

        except Exception as e:
            print(f"[Baseline MLP] ✗ Error: {e}")
            traceback.print_exc()
            return False
