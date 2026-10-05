# The synthetic NYISO archive for kind (CI). The generator runs at container start, so the
# archive's dates follow the day the pod starts. Build from the repo root:
#   docker build -f deploy/k8s/fixtures/archive.Dockerfile -t gridcast-archive-fixture:ci .
# archive.Dockerfile.dockerignore (BuildKit picks it up by name) admits only the five files
# copied below, so the image carries the generator and nothing else.
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.4.6 pandas==3.0.5
WORKDIR /fixture
COPY src/__init__.py src/config.py src/
COPY tests/__init__.py tests/synthetic.py tests/
COPY deploy/k8s/fixtures/make_archive.py ./
USER 10001
EXPOSE 8080
CMD ["python", "make_archive.py"]
