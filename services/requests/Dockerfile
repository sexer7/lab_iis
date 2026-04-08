FROM python:3.11-slim

COPY . /requests

WORKDIR /requests

RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt

CMD ["python", "requests.py"]

# docker build -t prediction-requests:1 .
# docker run --rm --network app-network -e PREDICTION_SERVICE_URL=http://ml_service:8000 prediction-requests:1
