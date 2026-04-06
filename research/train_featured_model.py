from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from common import (  # noqa: E402
    EXPERIMENT_NAME,
    REQUIREMENTS_FILE,
    TRACKING_URI,
    build_featured_preprocessor,
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
    X_train_fe_sklearn = split_data.X_train.copy()

    featured_preprocessor = build_featured_preprocessor()
    transformed_train = featured_preprocessor.fit_transform(X_train_fe_sklearn)
    feature_names = featured_preprocessor.get_feature_names_out()
    X_train_fe_sklearn = pd.DataFrame(
        transformed_train,
        columns=feature_names,
        index=split_data.X_train.index,
    )

    feature_names_path = SCRIPT_DIR / "featured_feature_names.txt"
    feature_names_path.write_text("\n".join(X_train_fe_sklearn.columns) + "\n", encoding="utf-8")

    pipeline = Pipeline(
        steps=[
            ("transform", featured_preprocessor),
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
    pipeline.fit(split_data.X_train, split_data.y_train)

    predictions = pipeline.predict(split_data.X_test)
    metrics = regression_metrics(split_data.y_test, predictions)
    signature = infer_signature(model_input=input_example, model_output=pipeline.predict(input_example))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="featured-random-forest") as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("feature_mode", "polynomial_and_binned_numeric_features")
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("numeric_transform", "StandardScaler+PolynomialFeatures+KBinsDiscretizer")
        mlflow.log_param("categorical_transform", "OrdinalEncoder")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("polynomial_degree", 2)
        mlflow.log_param("binned_features", "Year,Driven_kms")
        mlflow.log_param("binned_n_bins", 4)
        mlflow.log_metrics(metrics)

        summary = build_run_summary(dataset, split_data, metrics)
        summary_path = SCRIPT_DIR / "featured_metrics.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="reports")
        mlflow.log_artifact(feature_names_path, artifact_path="reports")
        mlflow.log_artifact(REQUIREMENTS_FILE, artifact_path="project_files")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="car-price-rf-featured",
            signature=signature,
            input_example=input_example,
            pip_requirements=load_requirements(),
        )

        print("MLflow run_id:", run.info.run_id)

    print("Featured model metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
