from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))

import mlflow
from mlflow import artifacts
from mlflow.tracking import MlflowClient


DEFAULT_RUN_ID = "6e05235bc5aa464db3d168d2249fbb42"
TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow' / 'mlflow.db').as_posix()}"
OUTPUT_DIR = ROOT_DIR / "services" / "models"
OUTPUT_PATH = OUTPUT_DIR / "model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a logged MLflow model by run_id into services/models/model.pkl",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="MLflow run_id of the model to download",
    )
    return parser.parse_args()


def resolve_model_source(client: MlflowClient, run_id: str) -> str:
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ValueError(f"No registered model version found for run_id={run_id}")
    return versions[0].source


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_source = resolve_model_source(client, args.run_id)
    download_root = OUTPUT_DIR / "_downloaded_model"
    if download_root.exists():
        shutil.rmtree(download_root)

    download_dir = Path(
        artifacts.download_artifacts(
            artifact_uri=model_source,
            dst_path=str(download_root),
        )
    )
    source_model_path = download_dir / "model.pkl"
    if not source_model_path.exists():
        raise FileNotFoundError(f"Downloaded model file not found: {source_model_path}")

    shutil.copy2(source_model_path, OUTPUT_PATH)
    shutil.rmtree(download_root, ignore_errors=True)

    print(f"run_id: {args.run_id}")
    print(f"model_source: {model_source}")
    print(f"saved_to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
