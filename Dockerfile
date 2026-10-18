FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.train.txt .
RUN pip install --no-cache-dir -r requirements.train.txt

COPY . .

ENTRYPOINT ["python", "train.py"]
