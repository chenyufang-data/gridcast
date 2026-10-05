#!/usr/bin/env bash
# End the GKE trial without leaving anything that bills.
#
#   deploy/k8s/gke/teardown.sh                    # keep the empty project and its budget
#   deploy/k8s/gke/teardown.sh --delete-project   # also delete them (the project: 30-day undo)
#
# Order matters: evidence first (it is gone with the cluster), then LoadBalancer Services and
# the namespace (the PVC's disk is deleted with it, reclaim policy Delete), then the cluster,
# then a check for orphaned disks, forwarding rules and addresses, then the registry.
# Deleting the cluster first can leave the persistent disk behind, still billing.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
G=(--project "$PROJECT" --quiet)
k() { kubectl --context "$CONTEXT" "$@"; }

if gcloud container clusters describe "$CLUSTER" --region="$REGION" "${G[@]}" >/dev/null 2>&1; then
  gcloud container clusters get-credentials "$CLUSTER" --region="$REGION" --project "$PROJECT"

  out="$ROOT/k8s-trial/$(date -u +%Y%m%dT%H%M%SZ)"
  step "evidence -> ${out#"$ROOT"/}"
  mkdir -p "$out"
  k -n "$NS" get all,pvc,cronjobs,jobs,configmaps -o wide >"$out/get.txt" 2>&1 || true
  k -n "$NS" get jobs --sort-by=.status.startTime \
    -o custom-columns='JOB:.metadata.name,START:.status.startTime,DONE:.status.completionTime,OK:.status.succeeded,FAILED:.status.failed' \
    >"$out/jobs.txt" 2>&1 || true
  k -n "$NS" rollout history deploy/gridcast-api >"$out/rollout-history.txt" 2>&1 || true
  k -n "$NS" get events --sort-by=.lastTimestamp >"$out/events.txt" 2>&1 || true
  k -n "$NS" top pods >"$out/top.txt" 2>&1 || true
  k -n "$NS" logs deploy/gridcast-api -c api --tail=2000 >"$out/api.log" 2>&1 || true
  k get nodes -o wide >"$out/nodes.txt" 2>&1 || true
  cat "$out/jobs.txt"

  step "LoadBalancer Services and the namespace"
  k get svc -A -o jsonpath='{range .items[?(@.spec.type=="LoadBalancer")]}{.metadata.namespace}{" "}{.metadata.name}{"\n"}{end}' |
    while read -r ns name; do
      [ -n "$name" ] && k -n "$ns" delete svc "$name" --wait=true
    done
  k delete namespace "$NS" --ignore-not-found --wait=true --timeout=10m

  step "cluster $CLUSTER"
  gcloud container clusters delete "$CLUSTER" --region="$REGION" "${G[@]}"
fi
kubectl config delete-context "$CONTEXT" >/dev/null 2>&1 || true

step "leftovers (each list must be empty)"
disks=$(gcloud compute disks list "${G[@]}" --filter="name~^pvc-" --format='value(name,zone.basename())' 2>/dev/null || true)
if [ -n "$disks" ]; then
  echo "orphaned volume disks, deleting:"
  echo "$disks"
  echo "$disks" | while read -r name zone; do gcloud compute disks delete "$name" --zone="$zone" "${G[@]}"; done
fi
echo "disks:            $(gcloud compute disks list "${G[@]}" --format='value(name)' 2>/dev/null | wc -l)"
echo "forwarding rules: $(gcloud compute forwarding-rules list "${G[@]}" --format='value(name)' 2>/dev/null | wc -l)"
echo "addresses:        $(gcloud compute addresses list "${G[@]}" --format='value(name)' 2>/dev/null | wc -l)"
echo "clusters:         $(gcloud container clusters list "${G[@]}" --format='value(name)' 2>/dev/null | wc -l)"

step "Artifact Registry $REPO"
if gcloud artifacts repositories describe "$REPO" --location="$REGION" "${G[@]}" >/dev/null 2>&1; then
  gcloud artifacts repositories delete "$REPO" --location="$REGION" "${G[@]}"
fi

if [ "${1:-}" = --delete-project ]; then
  step "budget and project"
  BILLING=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' | sed 's|billingAccounts/||')
  if [ -n "$BILLING" ]; then
    gcloud billing budgets list --billing-account="$BILLING" --billing-project="$PROJECT" \
      --filter="displayName='$BUDGET_NAME'" --format='value(name)' |
      while read -r budget; do gcloud billing budgets delete "$budget" --billing-project="$PROJECT" --quiet; done
  fi
  gcloud projects delete "$PROJECT" --quiet
fi
echo "done. The billing report for $PROJECT settles within a day: compare it with docs/k8s-plan.md section 5."
