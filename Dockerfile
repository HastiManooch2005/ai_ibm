FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    -r requirements.txt

COPY . .

ENV MLFLOW_ALLOW_FILE_STORE=true

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "runs:/c580cdf35b3841c7b818173afb27157b/model", "-p", "5001", "--host", "0.0.0.0"]