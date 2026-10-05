#!/usr/bin/env bash
# Create the 72-hour GKE Autopilot trial, from an empty project to a running stack:
#
#   BILLING=XXXXXX-XXXXXX-XXXXXX deploy/k8s/gke/up.sh      # gcloud billing accounts list
#
# Then: deploy/k8s/seed.sh gke, look at it through port-forward, and deploy/k8s/gke/teardown.sh
# within 72 hours. Every step checks first and skips what exists, so a failed run can simply
# be re-run. Laptop, Git Bash, Docker Desktop running (push.sh builds the images here).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${BILLING:?set BILLING to the billing account id (gcloud billing accounts list)}"

for tool in gcloud kubectl docker gke-gcloud-auth-plugin openssl; do
  command -v "$tool" >/dev/null || die "$tool not found (gke-gcloud-auth-plugin: gcloud components install gke-gcloud-auth-plugin)"
done
docker info >/dev/null 2>&1 || die "Docker is not running"
G=(--project "$PROJECT" --quiet)

step "project $PROJECT"
if ! gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  gcloud projects create "$PROJECT" --name="gridcast k8s trial"
fi
gcloud billing projects link "$PROJECT" --billing-account="$BILLING"
NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud services enable container.googleapis.com artifactregistry.googleapis.com \
  aiplatform.googleapis.com billingbudgets.googleapis.com "${G[@]}"

step "budget before anything that costs money"
# Alerts by e-mail at 50, 90 and 100 % of $15; a budget never stops spending, teardown does.
# Credits are excluded so the alerts track list cost even while the trial credit pays.
budget=$(gcloud billing budgets list --billing-account="$BILLING" --billing-project="$PROJECT" \
  --filter="displayName='$BUDGET_NAME'" --format='value(name)')
if [ -z "$budget" ]; then
  gcloud billing budgets create --billing-account="$BILLING" --billing-project="$PROJECT" \
    --display-name="$BUDGET_NAME" --budget-amount=15USD \
    --filter-projects="projects/$NUMBER" --credit-types-treatment=exclude-all-credits \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
fi

step "Artifact Registry $REGISTRY"
# a just-enabled API can answer PERMISSION_DENIED for a minute or two: retry
for attempt in 1 2 3 4 5 6 7 8; do
  gcloud artifacts repositories describe "$REPO" --location="$REGION" "${G[@]}" >/dev/null 2>&1 && break
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" \
    --immutable-tags --description="gridcast k8s trial images" "${G[@]}" && break
  [ "$attempt" -lt 8 ] || die "could not create the repository"
  echo "retrying in 30 s (the API may still be propagating)"
  sleep 30
done
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

step "images, pinned by digest in overlays/gke"
pinned() {  # every image of the overlay exists in this project's registry
  local ref
  for ref in $(kubectl kustomize "$OVERLAY" | sed -n "s|.*image: \($REGISTRY/.*@sha256:.*\)|\1|p" | sort -u); do
    gcloud artifacts docker images describe "$ref" --project "$PROJECT" >/dev/null 2>&1 || return 1
  done
  [ -n "${ref:-}" ]
}
if pinned; then echo "the pinned digests exist in $REGISTRY"; else "$ROOT/deploy/k8s/gke/push.sh" all; fi

step "Autopilot cluster $CLUSTER in $REGION (five to ten minutes)"
if ! gcloud container clusters describe "$CLUSTER" --region="$REGION" "${G[@]}" >/dev/null 2>&1; then
  gcloud container clusters create-auto "$CLUSTER" --region="$REGION" --release-channel=regular "${G[@]}"
fi
gcloud container clusters get-credentials "$CLUSTER" --region="$REGION" --project "$PROJECT"
k() { kubectl --context "$CONTEXT" "$@"; }

step "the nodes' service account may pull from the repository"
# explicit and narrow: a project without an organization usually grants it Editor anyway
gcloud artifacts repositories add-iam-policy-binding "$REPO" --location="$REGION" --project "$PROJECT" \
  --member="serviceAccount:$NUMBER-compute@developer.gserviceaccount.com" \
  --role=roles/artifactregistry.reader --format='value(etag)' >/dev/null

step "Workload Identity: Vertex AI for the UI's Kubernetes service account, no key"
# the pool <project>.svc.id.goog exists once the cluster does
gcloud projects add-iam-policy-binding "$PROJECT" --role=roles/aiplatform.user --condition=None \
  --member="principal://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/$PROJECT.svc.id.goog/subject/ns/$NS/sa/gridcast-ui" \
  --format='value(etag)' >/dev/null

step "namespace, admin Secret, apply"
k apply -f "$ROOT/deploy/k8s/base/namespace.yaml"
if ! k -n "$NS" get secret gridcast-admin >/dev/null 2>&1; then
  k -n "$NS" create secret generic gridcast-admin --from-literal=ADMIN_TOKEN="$(openssl rand -hex 24)"
fi
k diff -k "$OVERLAY" | head -60 || true
k apply -k "$OVERLAY"
# the first rollout waits for Autopilot to add nodes and pull 0.9 GB images
k -n "$NS" rollout status deploy/gridcast-api --timeout=900s
k -n "$NS" rollout status deploy/gridcast-ui --timeout=900s
k -n "$NS" get pods,pvc,cronjobs

step "up; tear down by $(date -d '+72 hours' '+%a %d %b %H:%M %Z')"
cat <<EOF
next:
  deploy/k8s/seed.sh gke                                              # about 5-15 minutes
  kubectl --context $CONTEXT -n $NS port-forward svc/gridcast-ui 8501:8501   # http://localhost:8501
  deploy/k8s/gke/teardown.sh                                          # within 72 hours
EOF
