from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
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
    load_selected_feature_indices,
    patch_mlflow_tempdir,
    prepare_workspace_temp_dir,
    regression_metrics,
    split_dataset,
)


TRIALS_COUNT = 10


def build_tunable_pipeline(
    selected_indices: list[int],
    *,
    n_estimators: int,
    max_depth: int,
    max_features: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("transform", build_featured_preprocessor()),
            ("select", FeatureIndexSelector(selected_indices)),
            (
                "regression",
                RandomForestRegressor(
                    random_state=42,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=2,
                    n_jobs=1,
                ),
            ),
        ]
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
    selected_indices = load_selected_feature_indices()

    selected_feature_names_path = SCRIPT_DIR / "selected_feature_names.txt"
    selected_feature_indices_path = SCRIPT_DIR / "selected_feature_indices.txt"

    # For regression, lower MAE is better, so the objective must be minimized.
    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 4, 20)
        max_features = trial.suggest_float("max_features", 0.1, 1.0)

        pipeline = build_tunable_pipeline(
            selected_indices,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
        )
        cv_scores = cross_val_score(
            pipeline,
            split_data.X_train,
            split_data.y_train,
            scoring="neg_mean_absolute_error",
            cv=3,
            n_jobs=1,
        )
        mae = -cv_scores.mean()
        trial.set_user_attr("mae", mae)
        return mae

    study = optuna.create_study(
        study_name="rf_featured_sfs_mae_tuning",
        direction="minimize",
    )
    study.optimize(objective, n_trials=TRIALS_COUNT, show_progress_bar=False)

    best_params = study.best_params
    best_pipeline = build_tunable_pipeline(
        selected_indices,
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
    )
    best_pipeline.fit(split_data.X_train, split_data.y_train)

    predictions = best_pipeline.predict(split_data.X_test)
    metrics = regression_metrics(split_data.y_test, predictions)
    signature = infer_signature(model_input=input_example, model_output=best_pipeline.predict(input_example))

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_path = SCRIPT_DIR / "tuning_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    best_params_path = SCRIPT_DIR / "best_tuned_params.txt"
    best_params_path.write_text(
        "\n".join(
            [
                "optimization_metric=mae",
                "optimization_direction=minimize",
                f"n_trials={TRIALS_COUNT}",
                f"best_mae_cv={study.best_value:.6f}",
                f"n_estimators={best_params['n_estimators']}",
                f"max_depth={best_params['max_depth']}",
                f"max_features={best_params['max_features']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="featured-random-forest-forward-sfs-optuna-tuned") as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("feature_mode", "polynomial_binned_features_with_forward_sfs")
        mlflow.log_param("tuning_library", "optuna")
        mlflow.log_param("optimization_metric", "mae")
        mlflow.log_param("optimization_direction", "minimize")
        mlflow.log_param("n_trials", TRIALS_COUNT)
        mlflow.log_param("selected_feature_count", len(selected_indices))
        mlflow.log_param("n_estimators", best_params["n_estimators"])
        mlflow.log_param("max_depth", best_params["max_depth"])
        mlflow.log_param("max_features", best_params["max_features"])
        mlflow.log_metric("best_cv_mae", study.best_value)
        mlflow.log_metrics(metrics)

        summary = build_run_summary(dataset, split_data, metrics)
        summary_path = SCRIPT_DIR / "tuned_model_metrics.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="reports")
        mlflow.log_artifact(trials_path, artifact_path="reports")
        mlflow.log_artifact(best_params_path, artifact_path="reports")
        mlflow.log_artifact(selected_feature_names_path, artifact_path="reports")
        mlflow.log_artifact(selected_feature_indices_path, artifact_path="reports")
        mlflow.log_artifact(REQUIREMENTS_FILE, artifact_path="project_files")

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            registered_model_name="car-price-rf-featured-sfs",
            signature=signature,
            input_example=input_example,
            pip_requirements=load_requirements(),
        )

        print("MLflow run_id:", run.info.run_id)

    print("Best params:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("Tuned model metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
