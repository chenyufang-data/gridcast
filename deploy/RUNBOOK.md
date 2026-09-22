# Deploy runbook — one GCE VM

The public demo runs on one **e2-small** VM (2 vCPU shared, 2 GB RAM + 2 GB swap, 20 GB
disk, static IPv4) in **us-east1**, with `docker-compose.prod.yml`: the backend, the
Streamlit frontend and Caddy (automatic HTTPS). Only ports 80 and 443 are public; SSH
goes through Google's IAP tunnel, so port 22 is never open to the internet. The chat
model is Gemini on Vertex AI, authenticated by the VM's service account: no key file
anywhere. Cost ≈ $19 per month (VM ≈ $12, disk ≈ $2, static IP ≈ $3, Vertex cents).

Laptop commands are **PowerShell** (run them from the repo root); commands on the VM
are bash after `gcloud compute ssh`. Values used throughout, change them if yours
differ:

| Placeholder | Value |
|---|---|
| project | `nyiso-gridcast` |
| region / zone | `us-east1` / `us-east1-b` |
| VM name | `gridcast` |
| service account | `gridcast-vm@nyiso-gridcast.iam.gserviceaccount.com` |
| domain | `gridcast.cyfang.org` |

## 0. Once, on the laptop

Install the Google Cloud SDK (`winget install Google.CloudSDK`, or the installer from
cloud.google.com/sdk), open a new PowerShell, then:

```powershell
gcloud auth login
gcloud config set project nyiso-gridcast
gcloud config set compute/region us-east1
gcloud config set compute/zone us-east1-b
```

Generate the admin token and keep it in your password manager; it is typed into the
VM's `.env` in step 4 and used for every write call afterwards:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 1. Project: APIs, service account, static IP, firewall (5 min)

```powershell
gcloud services enable compute.googleapis.com aiplatform.googleapis.com iap.googleapis.com
gcloud iam service-accounts create gridcast-vm --display-name "gridcast VM"
gcloud projects add-iam-policy-binding nyiso-gridcast `
  --member "serviceAccount:gridcast-vm@nyiso-gridcast.iam.gserviceaccount.com" `
  --role roles/aiplatform.user
gcloud compute addresses create gridcast-ip --region us-east1
gcloud compute addresses describe gridcast-ip --region us-east1 --format "value(address)"
```

Write the address down: it goes into the DNS record in step 7.

```powershell
# web from anywhere; SSH only from Google's IAP range; then close the default's open 22
gcloud compute firewall-rules create gridcast-web --network default --direction INGRESS `
  --allow "tcp:80,tcp:443" --target-tags gridcast --source-ranges 0.0.0.0/0
gcloud compute firewall-rules create gridcast-ssh-iap --network default --direction INGRESS `
  --allow tcp:22 --target-tags gridcast --source-ranges 35.235.240.0/20
gcloud compute firewall-rules delete default-allow-ssh --quiet
```

## 2. The VM (3 min, plus 2 min for the boot script)

```powershell
gcloud compute instances create gridcast `
  --machine-type e2-small `
  --image-family debian-12 --image-project debian-cloud `
  --boot-disk-size 20GB --boot-disk-type pd-balanced `
  --address gridcast-ip --tags gridcast `
  --service-account gridcast-vm@nyiso-gridcast.iam.gserviceaccount.com `
  --scopes cloud-platform `
  --metadata-from-file startup-script=deploy/bootstrap.sh
```

`--scopes cloud-platform` is what lets the containers reach Vertex AI with the service
account; without it the chat falls back to the keyword guide. The startup script
installs Docker, adds the swap file and creates `/opt/gridcast`. Watch it finish (the
first SSH also generates your key; answer the prompts):

```powershell
gcloud compute ssh gridcast --tunnel-through-iap --command "sudo journalctl -u google-startup-scripts -o cat -f"
```

Stop with Ctrl+C after `bootstrap done`, then verify:

```powershell
gcloud compute ssh gridcast --tunnel-through-iap --command "docker --version; free -m; df -h /"
```

## 3. Ship the code and the model bundle

The repo is private, so the VM never clones it: the laptop packs the committed tree and
copies it with the served TFT bundle over the tunnel.

```powershell
git archive --format=tar.gz -o gridcast.tar.gz HEAD
gcloud compute scp gridcast.tar.gz data/models/tft/tft.onnx data/models/tft/tft.json gridcast:/tmp/ --tunnel-through-iap
gcloud compute ssh gridcast --tunnel-through-iap
```

On the VM:

```bash
sudo tar -xzf /tmp/gridcast.tar.gz -C /opt/gridcast
cd /opt/gridcast && ls
```

Once the repo is public the same step is `sudo git clone https://github.com/chenyufang-data/gridcast /opt/gridcast`
and updates become `sudo git pull`.

## 4. `.env` on the VM (it never leaves the VM)

```bash
sudo tee /opt/gridcast/.env >/dev/null <<'EOF'
ADMIN_TOKEN=paste-the-token-from-step-0
SITE_ADDRESS=:80
LLM_PROVIDER=vertex
VERTEX_PROJECT=nyiso-gridcast
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.5-flash-lite
CHAT_LIMIT_PER_USER_DAY=20
CHAT_LIMIT_GLOBAL_DAY=300
LOG_LEVEL=INFO
EOF
sudo chmod 600 /opt/gridcast/.env
```

`SITE_ADDRESS=:80` serves plain HTTP on the bare IP until the DNS record exists (step 7).
Every other variable and its default is documented in `.env.sample`.

## 5. Build and start (10–15 min on e2-small)

```bash
cd /opt/gridcast
sudo docker compose -f docker-compose.prod.yml up -d --build
sudo docker compose -f docker-compose.prod.yml ps
curl -s http://127.0.0.1:8000/health | head -c 300; echo
curl -sI http://127.0.0.1/ | head -1
```

Expected: three containers `Up`, `"status":"ok"` from the backend, `HTTP/1.1 200 OK`
from Caddy. On the laptop, `http://<static ip>/` shows the overview with empty cards.
At first start the scheduler runs every job once (catch-up semantics): `forecast_all`
skips every zone on the empty store and `ingest_score` fetches the last 45 days; this
is expected and noisy in the backend log.

## 6. Bundle and seed (15–30 min, runs in the background)

```bash
sudo docker compose -f docker-compose.prod.yml exec backend mkdir -p /data/models/tft
sudo docker compose -f docker-compose.prod.yml cp /tmp/tft.onnx backend:/data/models/tft/tft.onnx
sudo docker compose -f docker-compose.prod.yml cp /tmp/tft.json backend:/data/models/tft/tft.json
curl -s http://127.0.0.1:8000/models; echo        # "loaded": true, "stale": false (the registry re-reads the folder)
nohup sudo docker compose -f docker-compose.prod.yml exec -T backend \
  python deploy/seed.py --months 15 --forecast-days 30 > /tmp/seed.log 2>&1 &
tail -f /tmp/seed.log
```

The seed ingests 15 months from the monthly zips (the 365-day training window plus
lags, the rolling 24-month retention grows the store from here), refreshes the weather,
forecasts the trailing 30 days plus tomorrow with the served TFT for every zone, and
scores them. It survives a dropped SSH session because of `nohup`. When it prints the
zone cards, `http://<static ip>/` shows real numbers.

## 7. Domain and HTTPS (5 min)

At the DNS provider of `cyfang.org`: an **A record** `gridcast` → the static IP, TTL 300.
Check from the laptop until it resolves:

```powershell
Resolve-DnsName gridcast.cyfang.org -Type A
```

Then on the VM:

```bash
sudo sed -i 's/^SITE_ADDRESS=.*/SITE_ADDRESS=gridcast.cyfang.org/' /opt/gridcast/.env
cd /opt/gridcast && sudo docker compose -f docker-compose.prod.yml up -d caddy
sudo docker compose -f docker-compose.prod.yml logs caddy --tail 30
```

Caddy obtains the Let's Encrypt certificate within a minute ("certificate obtained
successfully") and redirects HTTP to HTTPS. **Never delete the `caddy_data` volume**:
Let's Encrypt allows five certificates per domain per week, and the volume holds the
one you have.

## 8. The morning after

```bash
curl -s http://127.0.0.1:8000/jobs | python3 -m json.tool | head -40
```

`runs` shows `forecast_all` (04:30 ET), `ingest_score` (06:30 ET) and `isolf_refresh`
(08:30 ET) with `last_status: ok`; the overview cards show yesterday scored; the
Forecast view for tomorrow carries the TFT badge. Open the chat bubble and ask
anything: the caption in the window names the Gemini model and the replies left. If it
says "keyword guide", see Troubleshooting.

## 9. Snapshots, budget, idle

```powershell
gcloud compute resource-policies create snapshot-schedule gridcast-daily --region us-east1 `
  --max-retention-days 7 --start-time 09:00 --daily-schedule --storage-location us
gcloud compute disks add-resource-policies gridcast --resource-policies gridcast-daily --zone us-east1-b
```

Console → Billing → Budgets: a $25/month budget with e-mail alerts at 50, 90 and 100 %.
When the demo is idle, `gcloud compute instances stop gridcast` keeps the disk and the
static IP (≈ $5/month together) and `start` brings the site back in a minute; the
scheduler catches up the missed days on its own.

## 10. Operations

- **Logs.** `sudo docker compose -f docker-compose.prod.yml logs -f backend` (or
  `frontend`, `caddy`). Docker rotates them (5 × 20 MB per container).
- **Update after a commit.** Laptop: `git archive` + `gcloud compute scp` as in step 3
  (or `sudo git pull` once public). VM: `sudo tar -xzf /tmp/gridcast.tar.gz -C /opt/gridcast`
  then `sudo docker compose -f docker-compose.prod.yml up -d --build`. The data volume
  survives and the store migrates itself at startup. `sudo docker system prune -f`
  afterwards drops old image layers.
- **Monthly TFT refresh.** The bundle is served for 60 days after its fit cutoff
  (`TFT_MAX_AGE_DAYS`), then the trees take over and the sidebar says "stale"; refresh
  once a month. Laptop (torch venv, GPU): bring the data up to date, keep the old bundle
  for a rollback, export, copy:
  ```powershell
  .\.venv\Scripts\python.exe scripts\backfill.py
  .\.venv\Scripts\python.exe scripts\fetch_weather.py
  Copy-Item data\models\tft data\models\tft_<old fit date> -Recurse
  .\.venv\Scripts\python.exe scripts\export_tft.py     # ~2 min; verifies ONNX vs torch
  gcloud compute scp data\models\tft\tft.onnx data\models\tft\tft.json gridcast:/tmp/ --tunnel-through-iap
  ```
  VM: stage both files, then swap them in one step so the backend never sees a new model
  file next to an old manifest (the loader checks the SHA-256 and would reject the pair):
  ```bash
  cd /opt/gridcast
  sudo docker compose -f docker-compose.prod.yml exec backend mkdir -p /data/models/tft.new
  sudo docker compose -f docker-compose.prod.yml cp /tmp/tft.onnx backend:/data/models/tft.new/tft.onnx
  sudo docker compose -f docker-compose.prod.yml cp /tmp/tft.json backend:/data/models/tft.new/tft.json
  sudo docker compose -f docker-compose.prod.yml exec backend sh -c \
    'mv /data/models/tft.new/tft.onnx /data/models/tft/tft.onnx && mv /data/models/tft.new/tft.json /data/models/tft/tft.json && rmdir /data/models/tft.new'
  curl -s http://127.0.0.1:8000/models; echo     # new version string, "stale": false
  ```
  The registry re-reads the folder by itself; the next 04:30 ET job forecasts with the
  new bundle. Rollback = the same copy from the saved folder.
- **Admin calls from the laptop.** Tunnel the backend port, then use `/docs` with the
  `X-Admin-Token` header:
  `gcloud compute ssh gridcast --tunnel-through-iap -- -N -L 8000:127.0.0.1:8000`
  → `http://127.0.0.1:8000/docs`.
- **Teardown.** `gcloud compute instances delete gridcast`, then
  `gcloud compute addresses delete gridcast-ip --region us-east1` and the two firewall
  rules; or delete the whole project.

## 11. Troubleshooting

- **Chat answers as the keyword guide.** In order: the Vertex AI API is enabled in the
  project; the service account has `roles/aiplatform.user`; the VM has the
  `cloud-platform` scope (`gcloud compute instances describe gridcast --format "value(serviceAccounts[0].scopes)"`,
  a scope change needs a stop/start); `LLM_PROVIDER=vertex` and `VERTEX_PROJECT` are in
  `.env`. The frontend log names the error: `... logs frontend | grep -i vertex`.
- **Writes answer 503.** `ADMIN_TOKEN` is missing from `.env`; 401 means a wrong token.
- **The backend was killed during a tree fit** (`dmesg | grep -i oom`). The trees only
  train on demand; if it recurs, resize:
  `gcloud compute instances stop gridcast; gcloud compute instances set-machine-type gridcast --machine-type e2-medium; gcloud compute instances start gridcast`.
- **Seed says "no load data"**: the archive is unreachable from the VM;
  `curl -sI http://mis.nyiso.com/public/csv/pal/ | head -1` should be 200.
- **Certificate not issued.** The A record must resolve to the VM's IP from the public
  internet and port 80 must be open (Caddy answers the ACME challenge there).
- **`gcloud compute ssh` hangs.** The IAP firewall rule (step 1) or the
  `roles/iap.tunnelResourceAccessor` role is missing; project owners have it.
