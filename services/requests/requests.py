from __future__ import annotations

import itertools
import json
import os
import random
import time
from urllib import error, request


PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", "http://ml_service:8000").rstrip("/")
MIN_DELAY_SECONDS = float(os.getenv("MIN_DELAY_SECONDS", "0"))
MAX_DELAY_SECONDS = float(os.getenv("MAX_DELAY_SECONDS", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
ERROR_REQUEST_PROBABILITY = float(os.getenv("ERROR_REQUEST_PROBABILITY", "0.2"))

SAMPLE_OBJECTS = [
    {
        "Car_Name": "ritz",
        "Year": 2014,
        "Present_Price": 5.59,
        "Driven_kms": 27000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
    },
    {
        "Car_Name": "city",
        "Year": 2017,
        "Present_Price": 9.85,
        "Driven_kms": 14500,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
    },
    {
        "Car_Name": "innova",
        "Year": 2016,
        "Present_Price": 13.60,
        "Driven_kms": 52000,
        "Fuel_Type": "Diesel",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 1,
    },
    {
        "Car_Name": "corolla altis",
        "Year": 2015,
        "Present_Price": 14.79,
        "Driven_kms": 41000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Automatic",
        "Owner": 0,
    },
]


def build_payload() -> dict:
    payload = dict(random.choice(SAMPLE_OBJECTS))
    payload["Year"] = max(2003, payload["Year"] + random.randint(-1, 1))
    payload["Present_Price"] = round(max(0.5, payload["Present_Price"] * random.uniform(0.9, 1.1)), 2)
    payload["Driven_kms"] = max(500, int(payload["Driven_kms"] * random.uniform(0.8, 1.2)))
    payload["Owner"] = random.choice([0, 0, 1, 1, 2])
    return payload


def build_invalid_request(item_id: int) -> tuple[str, bytes]:
    bad_request_type = random.choice(["missing_field", "wrong_endpoint"])

    if bad_request_type == "wrong_endpoint":
        payload = build_payload()
        path = f"/api/predict/{item_id}"
    else:
        payload = build_payload()
        payload.pop("Fuel_Type", None)
        path = f"/api/prediction/{item_id}"

    return path, json.dumps(payload).encode("utf-8")


def send_prediction_request(item_id: int) -> None:
    if random.random() < ERROR_REQUEST_PROBABILITY:
        path, body = build_invalid_request(item_id)
    else:
        payload = build_payload()
        path = f"/api/prediction/{item_id}"
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url=f"{PREDICTION_SERVICE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response_body = json.loads(response.read().decode("utf-8"))

    print(
        f"[OK] item_id={item_id} status={response.status} "
        f"predict={response_body.get('predict')}"
    )


def main() -> None:
    print(f"Sending requests to {PREDICTION_SERVICE_URL}")
    print(f"Random delay range: {MIN_DELAY_SECONDS}..{MAX_DELAY_SECONDS} seconds")
    print(f"Error request probability: {ERROR_REQUEST_PROBABILITY}")

    try:
        for item_id in itertools.count(start=1):
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            time.sleep(delay)

            try:
                send_prediction_request(item_id)
            except (error.URLError, TimeoutError, ValueError) as exc:
                print(f"[ERROR] item_id={item_id} detail={exc}")
    except KeyboardInterrupt:
        print("Request sender stopped.")


if __name__ == "__main__":
    main()
