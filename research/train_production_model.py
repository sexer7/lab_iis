from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_NAME,
    FeatureIndexSelector,
    REQUIREMENTS_FILE,
    TRACKING_URI,
    TARGET_COLUMN,
    build_featured_preprocessor,
    load_clean_dataset,
    load_requirements,
    load_selected_feature_indices,
    patch_mlflow_tempdir,
    prepare_workspace_temp_dir,
)


BEST_EVALUATED_RUN_ID = "6d6108e1e17f425cb06afab37d7c53e0"
REGISTERED_MODEL_NAME = "car-price-rf-featured-sfs"


def main() -> None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prepare_workspace_temp_dir()
    patch_mlflow_tempdir()

    dataset = load_clean_dataset()
    input_example = dataset.drop(columns=[TARGET_COLUMN]).head(5).copy()
    selected_indices = load_selected_feature_indices()

    selected_feature_names_path = SCRIPT_DIR / "selected_feature_names.txt"
    selected_feature_indices_path = SCRIPT_DIR / "selected_feature_indices.txt"

    X_full = dataset.drop(columns=[TARGET_COLUMN]).copy()
    y_full = dataset[TARGET_COLUMN].copy()

    pipeline = Pipeline(
        steps=[
            ("transform", build_featured_preprocessor()),
            ("select", FeatureIndexSelector(selected_indices)),
            (
                "regression",
                RandomForestRegressor(
                    random_state=42,
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(X_full, y_full)

    signature = infer_signature(model_input=input_example, model_output=pipeline.predict(input_example))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="production-featured-random-forest-forward-sfs-full-data") as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("training_scope", "full_dataset")
        mlflow.log_param("feature_mode", "polynomial_binned_features_with_forward_sfs")
        mlflow.log_param("selected_feature_count", len(selected_indices))
        mlflow.log_param("best_evaluated_run_id", BEST_EVALUATED_RUN_ID)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("min_samples_leaf", 2)

        mlflow.log_artifact(REQUIREMENTS_FILE, artifact_path="project_files")
        mlflow.log_artifact(selected_feature_names_path, artifact_path="project_files")
        mlflow.log_artifact(selected_feature_indices_path, artifact_path="project_files")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example,
            pip_requirements=load_requirements(),
        )

        client = MlflowClient()
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            matched_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            version = max(
                (
                    mv.version
                    for mv in matched_versions
                    if mv.run_id == run.info.run_id
                ),
                key=int,
            )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=version,
            key="status",
            value="Production",
        )
        if hasattr(client, "set_registered_model_alias"):
            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias="Production",
                version=version,
            )

        print("MLflow run_id:", run.info.run_id)
        print("Production model version:", version)


if __name__ == "__main__":
    main()
