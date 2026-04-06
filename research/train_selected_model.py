from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlxtend.feature_selection import SequentialFeatureSelector
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
    feature_names = list(featured_preprocessor.get_feature_names_out())
    X_train_fe_sklearn = pd.DataFrame(
        transformed_train,
        columns=feature_names,
        index=split_data.X_train.index,
    )

    total_features = X_train_fe_sklearn.shape[1]
    selected_feature_count = max(5, round(total_features * 0.4))

    selector_estimator = RandomForestRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=1,
    )
    sfs = SequentialFeatureSelector(
        selector_estimator,
        k_features=selected_feature_count,
        forward=True,
        floating=False,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=1,
    )
    sfs.fit(X_train_fe_sklearn, split_data.y_train)

    selected_indices = list(sfs.k_feature_idx_)
    selected_feature_names = [feature_names[index] for index in selected_indices]

    selected_feature_names_path = SCRIPT_DIR / "selected_feature_names.txt"
    selected_feature_indices_path = SCRIPT_DIR / "selected_feature_indices.txt"
    selected_feature_names_path.write_text(
        "\n".join(selected_feature_names) + "\n",
        encoding="utf-8",
    )
    selected_feature_indices_path.write_text(
        "\n".join(str(index) for index in selected_indices) + "\n",
        encoding="utf-8",
    )

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
    pipeline.fit(split_data.X_train, split_data.y_train)

    predictions = pipeline.predict(split_data.X_test)
    metrics = regression_metrics(split_data.y_test, predictions)
    signature = infer_signature(model_input=input_example, model_output=pipeline.predict(input_example))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="featured-random-forest-forward-sfs") as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("feature_mode", "polynomial_binned_features_with_forward_sfs")
        mlflow.log_param("feature_selector", "mlxtend.SequentialFeatureSelector")
        mlflow.log_param("selector_direction", "forward")
        mlflow.log_param("selector_scoring", "neg_mean_absolute_error")
        mlflow.log_param("selector_cv", 3)
        mlflow.log_param("total_transformed_features", total_features)
        mlflow.log_param("selected_feature_count", selected_feature_count)
        mlflow.log_param("selected_feature_share", round(selected_feature_count / total_features, 4))
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_metrics(metrics)

        summary = build_run_summary(dataset, split_data, metrics)
        summary_path = SCRIPT_DIR / "selected_model_metrics.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="reports")
        mlflow.log_artifact(selected_feature_names_path, artifact_path="reports")
        mlflow.log_artifact(selected_feature_indices_path, artifact_path="reports")
        mlflow.log_artifact(REQUIREMENTS_FILE, artifact_path="project_files")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="car-price-rf-featured-sfs",
            signature=signature,
            input_example=input_example,
            pip_requirements=load_requirements(),
        )

        print("MLflow run_id:", run.info.run_id)

    print("Selected feature indices:")
    print(selected_indices)
    print("Selected feature names:")
    for feature_name in selected_feature_names:
        print(feature_name)

    print("Selected model metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
