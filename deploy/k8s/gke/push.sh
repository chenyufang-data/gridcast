#!/usr/bin/env bash
# Build the images, push them to the trial's Artifact Registry, and pin them by digest in
# overlays/gke. The model image's digest and its TFT_EXPECTED_VERSION change together, here.
#
#   deploy/k8s/gke/push.sh          # api, ui and model (the first deploy)
#   deploy/k8s/gke/push.sh model    # the monthly model refresh only
#
# Then read `git diff deploy/k8s/overlays/gke`, apply the overlay, and commit it: the overlay
# is the record of what runs. Tags are immutable in the repository; the digest deploys.
# Runs on the laptop (Git Bash, Docker Desktop): the bundle in data/models/tft is local only.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

what="${1:-all}"
case "$what" in all | model) ;; *) die "usage: push.sh [all|model]" ;; esac
BUNDLE="${BUNDLE:-$ROOT/data/models/tft}"
cd "$ROOT"

# pin <image name in the overlay> <registry image> <digest>: edits that entry of `images:`
pin() {
  sed -i -e "/^images:/,/^[a-zA-Z]/{/- name: $1\$/,/digest:/{s|newName: .*|newName: $2|;s|digest: .*|digest: $3|}}" \
    -e "s|VERTEX_PROJECT=.*|VERTEX_PROJECT=$PROJECT|" "$OVERLAY/kustomization.yaml"
}

# the digest the registry holds for image:tag, empty when there is none
remote_digest() {
  gcloud artifacts docker images describe "$1" --project "$PROJECT" \
    --format='value(image_summary.digest)' 2>/dev/null || true
}

# build_push <name> <tag> <docker build arguments...>: builds, pushes, prints the digest
build_push() {
  local ref="$REGISTRY/$1:$2"
  shift 2
  docker build --platform linux/amd64 -t "$ref" "$@" >&2
  docker push "$ref" >&2
  remote_digest "$ref"
}

step "model: verify the bundle with the serving code"
version=$("$PY" -c "import sys; from models.tft_onnx import OnnxTFT; print(OnnxTFT.load(sys.argv[1]).version)" "$BUNDLE") ||
  die "the bundle in $BUNDLE does not load"
sha12="${version#tft-onnx:}"
sha12="${sha12%%:*}"
echo "$version"
digest=$(remote_digest "$REGISTRY/model:tft-$sha12")
if [ -n "$digest" ]; then
  echo "already in the registry as model:tft-$sha12"
else
  digest=$(build_push model "tft-$sha12" -f deploy/k8s/model/Dockerfile "$BUNDLE")
fi
[ -n "$digest" ] || die "no digest for model:tft-$sha12"
pin gridcast-model "$REGISTRY/model" "$digest"
sed -i -e "s|TFT_EXPECTED_VERSION=.*|TFT_EXPECTED_VERSION=$version|" \
  -e "s|kubernetes.io/change-cause: \"model .*\"|kubernetes.io/change-cause: \"model $version\"|" \
  "$OVERLAY/kustomization.yaml"

if [ "$what" = all ]; then
  # the images are built from committed code only, so a digest always maps to a commit
  paths=(app frontend models src deploy/seed.py .streamlit Dockerfile Dockerfile.frontend
    .dockerignore requirements.lock requirements-frontend.lock)
  [ -z "$(git status --porcelain -- "${paths[@]}")" ] ||
    die "uncommitted changes in the image sources; commit them first"
  tag="$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)"

  step "api image $tag"
  digest=$(build_push api "$tag" .)
  [ -n "$digest" ] || die "no digest for api:$tag"
  pin gridcast-api "$REGISTRY/api" "$digest"

  step "ui image $tag"
  digest=$(build_push ui "$tag" -f Dockerfile.frontend .)
  [ -n "$digest" ] || die "no digest for ui:$tag"
  pin gridcast-ui "$REGISTRY/ui" "$digest"
fi

step "the overlay now pins"
kubectl kustomize "$OVERLAY" | grep -E "image: |TFT_EXPECTED_VERSION" | sort -u
git --no-pager diff --stat -- "$OVERLAY"
