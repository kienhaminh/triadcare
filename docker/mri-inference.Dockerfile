# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/mri-inference/requirements.txt /app/requirements.txt
# CPU-only torch: hippodeep runs one case in about 20 s on a CPU.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# hippodeep (Thyreau et al., 2018), code and trained weights, pinned to one upstream commit.
ADD https://github.com/bthyreau/hippodeep_pytorch.git#310305fe5a928b1da3e98e7c1384490881690593 \
    /app/third_party/hippodeep

COPY services/mri-inference/ /app/

# The DLBS NIfTI data is mounted read-only at runtime (docker compose), never baked into
# the image.

EXPOSE 8101

ENV MRI_HOST=0.0.0.0 \
    MRI_PORT=8101 \
    MRI_DATA_ROOT=/data/dlbs

CMD ["python", "app.py"]
