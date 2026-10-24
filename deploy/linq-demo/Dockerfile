FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/data/matplotlib YOLO_CONFIG_DIR=/data/ultralytics \
    TORCH_HOME=/data/torch XDG_CACHE_HOME=/data/cache \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/linq-demo/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY demo/ff/ /app/demo/ff/
COPY tools/run_presentation_demo.py tools/presentation_predict.py tools/fetch_artifact.py tools/check_demo.py /app/tools/
ARG MODEL_URL=https://github.com/Noizicrafty/GreatHanDetect/releases/download/models-2026-10-25/last.pt
ARG MODEL_SHA256=5b9df5d26785a3848643b7603036c2219dea4aedd6a19373dadaaa56182e519d
RUN python tools/fetch_artifact.py --url "$MODEL_URL" --sha256 "$MODEL_SHA256" --output /app/model/last.pt
RUN useradd --uid 10001 --create-home demo \
    && mkdir -p /data && chown demo:demo /data
USER 10001:10001
EXPOSE 8090
CMD ["python", "tools/run_presentation_demo.py", "--bind", "0.0.0.0", "--port", "8090", "--data", "/data", "--model-python", "/usr/local/bin/python", "--checkpoint", "/app/model/last.pt"]
