FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER requirements /tmp/requirements/

RUN micromamba install -y -n base -f /tmp/requirements/env_climada.yml && \
    micromamba update -y -n base -f /tmp/requirements/env_developer.yml && \
    micromamba clean --all --yes
