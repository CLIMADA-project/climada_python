# Start a AMD64 Ubuntu image
FROM --platform=linux/amd64 ubuntu:22.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Specify Miniconda version to download
ARG MINICONDA=Miniconda3-py39_22.11.1-1-Linux-x86_64.sh

# Install wget for downloading Miniconda
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/$MINICONDA \
    && mkdir /root/.conda \
    && bash $MINICONDA -b \
    && rm -f $MINICONDA
RUN conda init && bash

# Use mamba for faster installation
RUN conda install mamba -n base -c conda-forge

# Copy over the environment specs from repo
ADD requirements/env_climada.yml requirements/env_developer.yml ./

# Install requirements
RUN mamba env create -n climada_doc -f env_climada.yml \
    && mamba env update -n climada_doc -f env_developer.yml

# Default command
CMD ["/bin/bash"]
