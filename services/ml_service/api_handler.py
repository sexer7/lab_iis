from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys

import pandas as pd


SERVICE_DIR = Path(__file__).resolve().parent
 

def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "research").exists() and (candidate / "services").exists():
            return candidate
    return start_path


ROOT_DIR = find_project_root(SERVICE_DIR)
RESEARCH_DIR = ROOT_DIR / "research"
PROJECT_MODEL_PATH = ROOT_DIR / "services" / "models" / "model.pkl"
CONTAINER_MODEL_PATH = Path("/models/model.pkl")

for path in (SERVICE_DIR, RESEARCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)


def resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        return Path(configured_path)
    if CONTAINER_MODEL_PATH.exists():
        return CONTAINER_MODEL_PATH
    if CONTAINER_MODEL_PATH.parent.exists():
        return CONTAINER_MODEL_PATH
    return PROJECT_MODEL_PATH


class FastAPIHandler:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or resolve_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        with self.model_path.open("rb") as model_file:
            self.model = pickle.load(model_file)

    def predict(self, features: dict) -> float:
        payload = dict(features)
        payload["Owner"] = str(payload["Owner"])
        model_input = pd.DataFrame([payload])
        return float(self.model.predict(model_input)[0])
