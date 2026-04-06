from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_NAME,
    REQUIREMENTS_FILE,
    TRACKING_URI,
    build_baseline_pipeline,
    build_run_summary,
    load_clean_dataset,
    load_requirements,
    patch_mlflow_tempdir,
    prepare_workspace_temp_dir,
    regression_metrics,
    split_dataset,
)


def main() -> None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prepare_workspace_temp_dir()
    patch_mlflow_tempdir()

    dataset = load_clean_dataset()
    split_data = split_dataset(dataset)
    input_example = split_data.X_train.head(5).copy()

    pipeline = build_baseline_pipeline()
    pipeline.fit(split_data.X_train, split_data.y_train)

    predictions = pipeline.predict(split_data.X_test)
    metrics = regression_metrics(split_data.y_test, predictions)
    signature = infer_signature(model_input=input_example, model_output=pipeline.predict(input_example))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline-random-forest") as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("feature_mode", "baseline")
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("numeric_transform", "StandardScaler")
        mlflow.log_param("categorical_transform", "OrdinalEncoder")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metrics(metrics)

        summary = build_run_summary(dataset, split_data, metrics)
        summary_path = SCRIPT_DIR / "baseline_metrics.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="reports")
        mlflow.log_artifact(REQUIREMENTS_FILE, artifact_path="project_files")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="car-price-rf-baseline",
            signature=signature,
            input_example=input_example,
            pip_requirements=load_requirements(),
        )

        print("MLflow run_id:", run.info.run_id)

    print("Baseline metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
