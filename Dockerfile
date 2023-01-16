# Define the base image for the task
ARG base_image=mambaorg/micromamba:latest
FROM $base_image

# Copy the requirements into the container
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements /tmp/requirements/

# Install requirements, drop in an update for good measure
RUN micromamba install -y -n base -f /tmp/requirements/env_climada.yml && \
    micromamba install -y -n base -f /tmp/requirements/env_developer.yml && \
    micromamba update -y -n base && \
    micromamba clean --all --yes
