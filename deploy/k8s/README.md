# gridcast on Kubernetes

Kubernetes manifests for gridcast, next to the production setup (one VM with Docker
Compose, `deploy/RUNBOOK.md`), which they don't replace. A Kustomize base plus two overlays:
`kind` runs on a laptop and in CI, and `gke` is for a time-boxed GKE Autopilot trial in a
separate project.

| Path | What it is |
|---|---|
| `base/` | everything both environments share: the API and UI Deployments and Services, the data volume, ServiceAccounts, the NetworkPolicy, and the ConfigMap generated from `params.env` |
| `overlays/kind/` | locally built `:ci` images, no weather refresh, keyword chat; `kind-config.yaml` pins the node image (Kubernetes v1.36.4) |
| `overlays/gke/` | Artifact Registry images pinned by digest, a single-pod volume (`ReadWriteOncePod`), two UI replicas with session affinity, Vertex AI chat |

The daily jobs run as four CronJobs (`base/cronjobs.yaml`) in New York time, with
`concurrencyPolicy: Forbid`, a catch-up deadline and retries. Each run starts a small
trigger pod (`base/trigger.py`) that asks the API to run the job, so the API stays the only
writer, and the in-app scheduler thread is off (`SCHEDULER_ENABLED=0`). On kind the
schedules are suspended; run a job by hand:

```powershell
kubectl -n gridcast create job --from=cronjob/gridcast-ingest-score manual-1
kubectl -n gridcast logs -f job/manual-1
```

The decisions that shape it:

- **One writer for SQLite.** The API runs as exactly one replica with `strategy: Recreate`, and only it mounts the data volume. There is no autoscaler.
- **Probes.** `/livez` only shows that the process answers. `/readyz` checks a read-only database connection and, when `REQUIRE_TFT=1`, that the loaded model is the pinned `TFT_EXPECTED_VERSION`.
- **Measured resources.** API 250m CPU and 1 GiB memory; UI 100m and 512 MiB.
- **Hardened pods.** Non-root (uid 10001), read-only root filesystem, no service-account token mounted.
- **No secrets in git.** The admin token lives in the Secret `gridcast-admin`, created by hand.

## Run it on kind

PowerShell from the repo root, with Docker Desktop running and `kind` installed
(`winget install Kubernetes.kind`).

```powershell
kind create cluster --config deploy/k8s/overlays/kind/kind-config.yaml
docker build -t gridcast-api:ci .
docker build -f Dockerfile.frontend -t gridcast-ui:ci .
kind load docker-image --name gridcast gridcast-api:ci gridcast-ui:ci

kubectl apply -f deploy/k8s/base/namespace.yaml
$token = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
kubectl -n gridcast create secret generic gridcast-admin --from-literal=ADMIN_TOKEN=$token
kubectl apply -k deploy/k8s/overlays/kind
kubectl -n gridcast rollout status deploy/gridcast-api
kubectl -n gridcast rollout status deploy/gridcast-ui

kubectl -n gridcast port-forward svc/gridcast-ui 8501:8501     # http://localhost:8501
kubectl -n gridcast port-forward svc/gridcast-api 8000:8000    # http://localhost:8000/docs
```

The store starts empty. To fill it, ingest a range through the API (the token goes in
the `X-Admin-Token` header):

```powershell
curl.exe -X POST -H "X-Admin-Token: $token" "http://localhost:8000/admin/ingest?start=2026-09-14&end=2026-10-04"
```

Remove everything with `kind delete cluster --name gridcast`.

`tests/test_k8s_manifests.py` checks the invariants above on every test run: one API
replica, Recreate, no other pod on the volume, the scheduler off, requests and memory
limits on every container, no secrets.
