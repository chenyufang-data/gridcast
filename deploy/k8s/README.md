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
| `components/seed/`, `overlays/{kind,gke}-seed/` | the one-off seed: an environment with the API at zero replicas, the CronJobs suspended and a seed Job on the data volume |
| `seed.sh` | runs the seed on either environment and restores it afterwards |
| `gke/` | `up.sh`, `push.sh`, `teardown.sh` for the GKE trial (settings in `env.sh`) |

The daily jobs run as four CronJobs (`base/cronjobs.yaml`) in New York time, with
`concurrencyPolicy: Forbid`, a catch-up deadline and retries. Each run starts a small
trigger pod (`base/trigger.py`) that asks the API to run the job, so the API stays the only
writer, and the in-app scheduler thread is off (`SCHEDULER_ENABLED=0`). On kind the
schedules are suspended; run a job by hand:

```powershell
kubectl -n gridcast create job --from=cronjob/gridcast-ingest-score manual-1
kubectl -n gridcast logs -f job/manual-1
```

The served model ships as its own tiny image (`model/Dockerfile`: busybox plus `tft.onnx`
and `tft.json`). An init container copies it into the API pod, never onto the data
volume. The overlays reference it by digest and pin its identity in the `gridcast-model`
ConfigMap, which only the API reads (`TFT_EXPECTED_VERSION`, with `REQUIRE_TFT=1`). A pod
whose bundle doesn't match never becomes ready. A model rollout is an ordinary
Deployment rollout, and `kubectl rollout undo deploy/gridcast-api` restores the previous
model with its pin. kind serves `fixtures/tft-tiny`, a 185 KiB bundle trained on
synthetic data by `fixtures/make_tiny_bundle.py`:

```powershell
docker build -f deploy/k8s/model/Dockerfile -t gridcast-model:fixture deploy/k8s/fixtures/tft-tiny
kind load docker-image --name gridcast gridcast-model:fixture
```

The decisions that shape it:

- **One writer for SQLite.** The API runs as exactly one replica with `strategy: Recreate`, and only it mounts the data volume. There is no autoscaler.
- **Probes.** `/livez` only shows that the process answers. `/readyz` checks a read-only database connection and, when `REQUIRE_TFT=1`, that the loaded model is the pinned `TFT_EXPECTED_VERSION`.
- **Measured resources.** API 250m CPU and 1 GiB memory; UI 100m and 512 MiB.
- **Hardened pods.** Non-root (uid 10001), read-only root filesystem, no service-account token mounted.
- **The writer outranks the rest.** The API and the seed Job use the PriorityClass `gridcast-writer` (`base/priorityclass.yaml`). On a full node a system pod evicts the lowest priority first; on the GKE trial, before this class, that was the API, 36 times in 46 hours.
- **No secrets in git.** The admin token lives in the Secret `gridcast-admin`, created by hand.

## Run it on kind

PowerShell from the repo root, with Docker Desktop running and `kind` installed
(`winget install Kubernetes.kind`). This is the same sequence CI runs
(`.github/workflows/k8s.yml`), about a minute once the images exist.

```powershell
kind create cluster --config deploy/k8s/overlays/kind/kind-config.yaml
docker build -t gridcast-api:ci .
docker build -f Dockerfile.frontend -t gridcast-ui:ci .
docker build -f deploy/k8s/fixtures/archive.Dockerfile -t gridcast-archive-fixture:ci .
docker build -f deploy/k8s/model/Dockerfile -t gridcast-model:fixture deploy/k8s/fixtures/tft-tiny
kind load docker-image --name gridcast gridcast-api:ci gridcast-ui:ci gridcast-archive-fixture:ci gridcast-model:fixture

kubectl apply -f deploy/k8s/base/namespace.yaml
$token = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
kubectl -n gridcast create secret generic gridcast-admin --from-literal=ADMIN_TOKEN=$token
kubectl apply -k deploy/k8s/overlays/kind
kubectl -n gridcast rollout status deploy/gridcast-api

.\.venv\Scripts\python.exe deploy/k8s/ci/smoke.py            # the CI checks, against this cluster
kubectl -n gridcast port-forward svc/gridcast-ui 8501:8501     # http://localhost:8501
```

On kind the API reads a synthetic NYISO archive served inside the cluster
(`fixtures/make_archive.py`, through `NYISO_ARCHIVE_BASE`), written at pod start for the
last two to three months, so nothing leaves the cluster. The smoke test fills the store
by running the ingest job once. Remove everything with `kind delete cluster --name gridcast`.

`tests/test_k8s_manifests.py` checks the invariants above on every test run: one API
replica, Recreate, no other pod on the volume, the scheduler off, requests and memory
limits on every container, no secrets.

## Seed the store

A new volume is empty. `deploy/seed.py` fills it (the archive, weather, 30 days of
forecasts and their scores), and it writes the database, so it runs only while the API is
stopped:

```bash
deploy/k8s/seed.sh kind     # or gke
```

The script scales the API to zero and waits for its pod to go, then applies
`overlays/<env>-seed`. That overlay is the environment plus `components/seed`: the API
stays at zero, every CronJob is suspended, and the Job `gridcast-seed` runs on the volume
with the API's own image, model and settings (copied from the rendered Deployment, so
`push.sh` keeps editing one file). When the Job ends, the script re-applies the plain
overlay; `kubectl apply` drops the fields the seed set, so the API returns and the
schedules resume. On GKE the volume is `ReadWriteOncePod`, so a seed pod started next to a
running API would stay `Pending`. The test suite checks that a seed overlay differs from
its environment in exactly those fields.

Measured on kind against the real archive (15 months, 30 forecast days, the real TFT):
149 s and a cgroup memory peak of 425 MiB, so the Job gets 1 GiB. The last line of the
Job's log prints the peak again on every run.

## The GKE trial

A time-boxed trial on GKE Autopilot in its own project (`nyiso-gridcast-k8s`), so it
cannot touch the production VM. Access is `kubectl port-forward` only: no public IP, no
DNS. Expected cost about $2.30 for 72 hours. Run from the laptop in Git Bash, with Docker
Desktop running and `gcloud components install gke-gcloud-auth-plugin` done once.

```bash
BILLING=XXXXXX-XXXXXX-XXXXXX deploy/k8s/gke/up.sh    # gcloud billing accounts list
deploy/k8s/seed.sh gke
kubectl --context gke_nyiso-gridcast-k8s_us-east1_gridcast -n gridcast port-forward svc/gridcast-ui 8501:8501
deploy/k8s/gke/teardown.sh                            # within 72 hours
```

`up.sh` creates the project and links billing, enables the APIs, creates a $15 budget with
alerts at 50, 90 and 100 % (credits excluded, so the alerts track list cost) before
anything bills, creates an Artifact Registry repository with immutable tags, runs
`push.sh`, creates the Autopilot cluster, grants Vertex AI to the UI's Kubernetes service
account through Workload Identity (no Google service account, no key), creates the
namespace and the admin Secret, and applies the overlay. Each step skips what already
exists, so a failed run can be re-run. No script changes the gcloud default project, and
every `kubectl` call names the trial cluster's context.

`push.sh` loads the model bundle with the serving code first, then builds and pushes the
images and writes their digests into `overlays/gke`, together with the model's
`TFT_EXPECTED_VERSION`. The API and UI images are built only from committed code, tagged
with the commit. The monthly model refresh is `push.sh model`, then `kubectl apply -k`.

`teardown.sh` saves the evidence (jobs, rollout history, events, logs) under `k8s-trial/`
(gitignored), deletes any LoadBalancer Service and the namespace (the volume's disk goes
with its claim), then the cluster, deletes orphaned `pvc-*` disks, lists forwarding rules
and addresses (all must be zero), and deletes the registry. `--delete-project` also
removes the budget and the project.
