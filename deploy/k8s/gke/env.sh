# shellcheck shell=bash disable=SC2034  # sourced: the variables are used by the scripts
# Settings shared by the GKE trial scripts (sourced, not run). Override any of them in the
# environment, e.g. PROJECT=nyiso-gridcast-k8s-2 if the project ID is taken.
# No script here runs `gcloud config set`: the live site's project stays the gcloud
# default, and every command names the trial project and the trial cluster's context.

PROJECT="${PROJECT:-nyiso-gridcast-k8s}"
REGION="${REGION:-us-east1}"
CLUSTER="${CLUSTER:-gridcast}"
REPO="${REPO:-gridcast}"
BUDGET_NAME="${BUDGET_NAME:-gridcast k8s trial}"

REGISTRY="$REGION-docker.pkg.dev/$PROJECT/$REPO"
CONTEXT="gke_${PROJECT}_${REGION}_${CLUSTER}"
NS=gridcast

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OVERLAY="$ROOT/deploy/k8s/overlays/gke"

# the repo's virtual environment on the laptop (Windows or POSIX layout), else the system's
for PY in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" python3 python; do
  command -v "$PY" >/dev/null 2>&1 && break
done

step() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
