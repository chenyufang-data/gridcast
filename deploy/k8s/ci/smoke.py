"""Post-deploy smoke test for the kind cluster, in CI and on a laptop.

    python deploy/k8s/ci/smoke.py

Needs kubectl pointed at the cluster and the kind overlay applied (synthetic archive,
fixture model). Standard library only, so it runs the same on the GitHub runner and on
Windows. Checks, in order:

1. /readyz answers 200 and the served TFT is exactly the pinned version;
2. the ingest CronJob, run once by hand, completes and leaves yesterday ingested;
3. a forecast of yesterday for N.Y.C. is served by the pinned TFT and scored (on creation,
   or earlier by a seed that made the same forecast);
4. a tree retrain of yesterday for WEST is stored and scored;
5. the UI answers its health check and serves its page.
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

NS = "gridcast"
API, UI = 18000, 18501


def kubectl(*args: str) -> str:
    done = subprocess.run(["kubectl", "-n", NS, *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"kubectl {' '.join(args)} failed:\n{done.stderr}")
    return done.stdout


def port_forward(service: str, local: int, remote: int) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        ["kubectl", "-n", NS, "port-forward", f"svc/{service}", f"{local}:{remote}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", local)) == 0:
                return proc
        time.sleep(0.5)
    proc.terminate()
    raise SystemExit(f"port-forward to {service} did not open")


def call(method: str, path: str, port: int = API, token: str | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    if token:
        request.add_header("X-Admin-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read().decode(), exc.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def score_of(fc: dict[str, Any]) -> dict[str, Any]:
    """The forecast's score: in the answer when it was just created, else as stored (after a seed)."""
    if fc.get("new") is not False:
        return fc.get("score") or {}
    zone = fc["zone"].lower().replace(".", "").replace(" ", "_")
    _, stored = call(
        "GET", f"/zones/{zone}/forecasts/{fc['target_date']}?version={fc['model_version']}"
    )
    return stored.get("score") or {}


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"   ok: {message}", flush=True)


def main() -> None:
    token = base64.b64decode(
        kubectl("get", "secret", "gridcast-admin", "-o", "jsonpath={.data.ADMIN_TOKEN}")
    ).decode()
    yesterday = (datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)).isoformat()
    forwards = [port_forward("gridcast-api", API, 8000), port_forward("gridcast-ui", UI, 8501)]
    try:
        step("1. readiness with the pinned model")
        status, ready = call("GET", "/readyz")
        print(f"   {ready}")
        require(status == 200, "/readyz answers 200")
        pin = ready["pinned_version"]
        require(bool(pin) and ready["model_version"] == pin, f"the served TFT is the pin {pin}")

        step("2. the ingest CronJob, run once")
        job = f"smoke-ingest-{int(time.time())}"
        kubectl("create", "job", "--from=cronjob/gridcast-ingest-score", job)
        waited = subprocess.run(
            [
                "kubectl",
                "-n",
                NS,
                "wait",
                "--for=condition=complete",
                f"job/{job}",
                "--timeout=300s",
            ],
            capture_output=True,
            text=True,
        )
        print("   " + kubectl("logs", f"job/{job}").strip()[:300])
        require(waited.returncode == 0, f"job {job} completed")
        _, health = call("GET", "/health")
        last = health["data"]["load_slots"]["last_ingested_day"]
        require(last == yesterday, f"load ingested through yesterday ({last})")

        step("3. a forecast of yesterday for N.Y.C., served by the pinned TFT")
        status, fc = call("POST", f"/zones/nyc/forecasts?target_date={yesterday}", token=token)
        require(status == 200, f"forecast stored (HTTP {status})")
        score = score_of(fc)
        print(
            f"   model {fc['model']}, {len(fc['values'])} slots, MAPE {score.get('mape_hour')} "
            "(tiny synthetic fixture: this checks the plumbing, not accuracy)"
        )
        require(fc["model"] == pin, "N.Y.C. served by the pinned TFT")
        require(len(fc["values"]) >= 92, "a full day of 15-minute slots")
        require(score.get("coverage") == 1.0, "scored against the day's actuals")

        step("4. a tree retrain of yesterday for WEST")
        status, lg = call(
            "POST", f"/zones/west/forecasts?target_date={yesterday}&model=lgbm", token=token
        )
        require(status == 200, f"retrain stored (HTTP {status})")
        require(lg["model"].startswith("lgbm:"), f"trees served ({lg['model']})")
        require(score_of(lg).get("coverage") == 1.0, "scored against the day's actuals")

        step("5. the UI")
        status, body = call("GET", "/_stcore/health", port=UI)
        require(status == 200 and body == "ok", "Streamlit health is ok")
        status, _ = call("GET", "/", port=UI)
        require(status == 200, "the page is served")
    finally:
        for proc in forwards:
            proc.terminate()
    print("\nsmoke test passed")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc.code, file=sys.stderr)
            sys.exit(1)
        raise
