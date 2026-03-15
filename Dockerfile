# Build:  docker build -t model-home/climada_python:latest .
# Run:    echo '{}' | docker run --rm -i model-home/climada_python:latest
# With input file mounted:
#   docker run --rm -v "$PWD/run:/run" model-home/climada_python:latest /run/hurricane_scenario.json

# micromamba is the fastest conda solver and handles all C-level geospatial deps
FROM mambaorg/micromamba:1.5-jammy

USER root
WORKDIR /app

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install all heavy geospatial dependencies via conda-forge
# (gdal, cartopy, fiona, rasterio etc. require compiled C libs — conda handles this cleanly)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml ./
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean -ya

# Install CLIMADA from local source (editable so it sees the climada/ package)
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY climada ./climada
COPY runner.py ./

# --no-deps: all deps are already handled by conda; pip only wires the local package
RUN micromamba run -n base pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "runner.py"]
# Default: read from stdin (empty JSON → all defaults → Cat 4 Miami)
CMD ["-"]
