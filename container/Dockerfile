FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/opt/ml/model

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git ffmpeg libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone SadTalker
RUN git clone https://github.com/OpenTalker/SadTalker.git /opt/sadtalker
WORKDIR /opt/sadtalker

# Install SadTalker dependencies
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    face_alignment==1.3.5 \
    imageio==2.19.3 \
    imageio-ffmpeg==0.4.7 \
    librosa==0.9.2 \
    numba \
    resampy==0.3.1 \
    pydub==0.25.1 \
    scipy==1.10.1 \
    kornia==0.6.8 \
    tqdm \
    yacs==0.1.8 \
    pyyaml \
    joblib==1.1.0 \
    scikit-image==0.19.3 \
    basicsr==1.4.2 \
    facexlib==0.3.0 \
    gfpgan==1.3.8 \
    av \
    safetensors \
    flask \
    gunicorn \
    boto3

# Download checkpoints
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vinthony/SadTalker-V002rc', local_dir='./checkpoints', local_dir_use_symlinks=False)"

# Copy inference server
COPY serve.py /opt/sadtalker/serve.py

EXPOSE 8080

ENTRYPOINT ["python", "serve.py"]
