FROM python:3.11-slim

COPY . /ml_service

VOLUME ["/models"]

WORKDIR /ml_service

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t car-price-service:1 .
# docker run --rm -p 8000:8000 -v "$(pwd)/../models:/models" car-price-service:1
# PowerShell: docker run --rm -p 8000:8000 -v "$((Resolve-Path ..\models).Path):/models" car-price-service:1
