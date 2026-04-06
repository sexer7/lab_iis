from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MLFLOW_DIR = ROOT_DIR / "mlflow"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "car-price-regression"
TARGET_COLUMN = "Selling_Price"
NUMERIC_FEATURES = ["Year", "Present_Price", "Driven_kms"]
CATEGORICAL_FEATURES = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission", "Owner"]
POLYNOMIAL_FEATURES = ["Year", "Present_Price", "Driven_kms"]
BINNED_FEATURES = ["Year", "Driven_kms"]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class FeatureIndexSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_indices: list[int] | tuple[int, ...]):
        self.selected_indices = selected_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indices = list(self.selected_indices)
        if hasattr(X, "iloc"):
            return X.iloc[:, indices]
        return X[:, indices]


def load_clean_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATA_DIR / "car_data_cleaned.csv")
    for column in CATEGORICAL_FEATURES:
        dataset[column] = dataset[column].astype(str)
    return dataset


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> SplitData:
    X = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = dataset[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_baseline_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = build_baseline_preprocessor()

    model = RandomForestRegressor(random_state=random_state, n_estimators=100, n_jobs=1)

    return Pipeline(
        steps=[
            ("transform", preprocessor),
            ("regression", model),
        ]
    )


def build_baseline_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        )
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
    )


def build_featured_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = build_featured_preprocessor()

    model = RandomForestRegressor(
        random_state=random_state,
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("transform", preprocessor),
            ("regression", model),
        ]
    )


def build_featured_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num_scaled", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "num_poly",
                Pipeline(
                    [
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                POLYNOMIAL_FEATURES,
            ),
            (
                "num_bins",
                KBinsDiscretizer(
                    n_bins=4,
                    encode="onehot-dense",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
                BINNED_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        )
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
    )


def regression_metrics(y_true: pd.Series, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mape": float(mean_absolute_percentage_error(y_true, predictions)),
        "mse": float(mean_squared_error(y_true, predictions)),
    }


def build_run_summary(
    dataset: pd.DataFrame,
    split_data: SplitData,
    metrics: dict[str, float],
) -> pd.DataFrame:
    summary = {
        "dataset_rows": len(dataset),
        "train_rows": len(split_data.X_train),
        "test_rows": len(split_data.X_test),
        **metrics,
    }
    return pd.DataFrame([summary])


def prepare_workspace_temp_dir() -> Path:
    temp_dir = ROOT_DIR / ".tmp" / "mlflow_runtime"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    return temp_dir


def prepare_model_export_dir(model_name: str) -> Path:
    export_dir = ROOT_DIR / ".tmp" / "exported_models" / model_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def load_requirements() -> list[str]:
    return [
        line.strip()
        for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_selected_feature_indices(path: Path | None = None) -> list[int]:
    indices_path = path or (ROOT_DIR / "research" / "selected_feature_indices.txt")
    return [int(line.strip()) for line in indices_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def patch_mlflow_tempdir() -> Path:
    import mlflow.utils.file_utils as mlflow_file_utils

    base_dir = ROOT_DIR / ".tmp" / "mlflow_model_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)

    def workspace_create_tmp_dir() -> str:
        temp_dir = base_dir / f"tmp_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        return str(temp_dir)

    mlflow_file_utils.create_tmp_dir = workspace_create_tmp_dir
    return base_dir
