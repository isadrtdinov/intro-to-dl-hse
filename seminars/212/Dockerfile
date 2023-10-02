ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.2.0
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER}

WORKDIR /srv/www/hse-dl/

COPY . .

ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update --yes --quiet && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common build-essential apt-utils wget curl git && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet

RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils pip

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.6.1 python3 - && poetry install

CMD ["jupyter", "lab", "--FileContentsManager.delete_to_trash=False", "--allow-root", "--port=44444", "--ip=0.0.0.0"]