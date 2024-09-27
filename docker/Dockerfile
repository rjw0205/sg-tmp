FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install opeslide-tools
RUN apt-get update && \
    apt-get install -y \
    openslide-tools \
    tmux \
    git \
    wget \
    curl \
    vim

# Install zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.0/zsh-in-docker.sh)" -- \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

# Install Google Cloud CLI
RUN curl https://sdk.cloud.google.com > install.sh && \
    bash install.sh --disable-prompts --install-dir=/ && \
    rm install.sh

# Install Python dependencies using pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
