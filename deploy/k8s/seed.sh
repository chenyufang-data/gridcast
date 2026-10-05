#!/usr/bin/env bash
# Seed the store once, with the API stopped (SQLite has one writer):
#
#   deploy/k8s/seed.sh kind     # rehearsal on the laptop's kind cluster (synthetic archive)
#   deploy/k8s/seed.sh gke      # the GKE trial: 15 months of the real archive
#
# 1. scale the API to zero and wait until its pod is gone (kind has no ReadWriteOncePod);
# 2. apply overlays/<env>-seed: the API stays at zero, the CronJobs are suspended, the
#    seed Job starts on the data volume;
# 3. follow the Job to its end, success or failure;
# 4. re-apply overlays/<env>: the API returns and the schedules resume.
# Re-running is safe: the seed upserts, and an earlier seed Job is deleted first.
set -euo pipefail

env="${1:-}"
K8S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$env" in
  kind) CONTEXT=kind-gridcast ;;
  gke) . "$K8S/gke/env.sh" ;;
  *) echo "usage: seed.sh kind|gke" >&2; exit 2 ;;
esac
k() { kubectl --context "$CONTEXT" -n gridcast "$@"; }

echo "== stop the API"
k delete job gridcast-seed --ignore-not-found
k scale deploy/gridcast-api --replicas=0
k wait --for=delete pod -l app.kubernetes.io/component=api --timeout=180s

echo "== start the seed"
kubectl --context "$CONTEXT" apply -k "$K8S/overlays/$env-seed"
k logs -f job/gridcast-seed -c seed --pod-running-timeout=15m || true

state=""
until [ -n "$state" ]; do
  state=$(k get job gridcast-seed \
    -o jsonpath='{.status.conditions[?(@.type=="Complete")].type}{.status.conditions[?(@.type=="Failed")].type}')
  [ -n "$state" ] || sleep 10
done

echo "== seed $state; bring the API back"
kubectl --context "$CONTEXT" apply -k "$K8S/overlays/$env"
k rollout status deploy/gridcast-api --timeout=600s
k get cronjobs
[ "$state" = Complete ]
