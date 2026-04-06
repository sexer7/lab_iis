from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from services.ml_service.api_handler import FastAPIHandler
except ModuleNotFoundError:
    from api_handler import FastAPIHandler


app = FastAPI(
    title="Car Price ML Service",
    version="0.1.0",
    description="Service scaffold for LR3 deployment of the production car price model.",
)


SERVICE_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for candidate in (start_path, *start_path.parents):
        if (candidate / "research").exists() and (candidate / "services").exists():
            return candidate
    return start_path


ROOT_DIR = find_project_root(SERVICE_DIR)
RESEARCH_DIR = ROOT_DIR / "research"
if RESEARCH_DIR.exists() and str(RESEARCH_DIR) not in sys.path:
    sys.path.append(str(RESEARCH_DIR))


class CarFeatures(BaseModel):
    Car_Name: str = Field(..., examples=["ritz"])
    Year: int = Field(..., examples=[2014])
    Present_Price: float = Field(..., examples=[5.59])
    Driven_kms: int = Field(..., examples=[27000])
    Fuel_Type: str = Field(..., examples=["Petrol"])
    Selling_type: str = Field(..., examples=["Dealer"])
    Transmission: str = Field(..., examples=["Manual"])
    Owner: int | str = Field(..., examples=[0])


@lru_cache(maxsize=1)
def get_handler() -> FastAPIHandler:
    return FastAPIHandler()


@app.get("/")
def root() -> dict[str, str]:
    return {"Hello": "World"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/prediction/{item_id}")
def predict(item_id: int, features: CarFeatures) -> dict[str, float | int]:
    try:
        handler = get_handler()
        prediction = handler.predict(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {"item_id": item_id, "predict": prediction}
