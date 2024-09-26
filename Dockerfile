# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install general build dependencies and useful tools
RUN apt-get update && apt-get install -y --no-install-recommends  \
    apt-utils \
    bash \
    bash-completion \
    build-essential \
    curl \
    git \
    htop \
    less \
    locales \
    make \
    nano \
    openslide-tools \
    pkg-config \
    psmisc \
    rsync \
    screen \
    software-properties-common \
    ssh \
    sudo \
    tar \
    tmux \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    zip \
    zsh

# Install the gcloud CLI. https://cloud.google.com/sdk/docs/install#installation_instructions.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y \
    && apt-get install google-cloud-cli -y

# Install python3.10
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install python packages via pip
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# PyTorch for CUDA 11.8
RUN pip --no-cache-dir install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Set up the working directory in the container
WORKDIR /workspace

# Copy your local repository into the Docker container
COPY . /workspace

# Install Python dependencies using pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Set PATH
ENV PATH="${PATH}:~/.local/bin"
ENV PYTHONPATH="${PYTHONPATH}:/workspace"