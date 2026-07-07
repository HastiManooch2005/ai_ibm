FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "runs:/c580cdf35b3841c7b818173afb27157b/model", "-p", "5001", "--host", "0.0.0.0"]