# CPU-only image for inference. Build: docker build -t pad .
# Run:   docker run --rm -v /path/to/images:/data:ro -v "$PWD":/out pad --input /data --output /out/outputs.csv
FROM python:3.9-slim

# libGL/libglib are needed by opencv-python-headless at import; libheif by pillow-heif
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libheif1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-infer.txt .
RUN pip install --no-cache-dir -r requirements-infer.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY pad/ pad/
COPY artefacts/config.json artefacts/model.pt artefacts/
COPY infer.py .

# a mounted input folder and a mounted output folder; defaults write outputs.csv to /out
VOLUME ["/data", "/out"]
ENV OMP_NUM_THREADS=4
ENTRYPOINT ["python", "infer.py"]
CMD ["--input", "/data", "--output", "/out/outputs.csv"]
