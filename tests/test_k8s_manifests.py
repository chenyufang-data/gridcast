"""Invariants of the Kubernetes manifests under deploy/k8s (docs/k8s-plan.md).

The rendered overlays are checked with `kubectl kustomize` when kubectl is installed (it
is on the GitHub runners and the laptop); the secret scan reads the raw files and always
runs. These encode the decisions that keep the deployment safe: one writer for SQLite,
the in-app scheduler off, measured resources on every container, no secrets in git.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

K8S = Path(__file__).resolve().parent.parent / "deploy" / "k8s"
OVERLAYS = ("kind", "gke")
SEED_OVERLAYS = ("kind-seed", "gke-seed")  # an environment plus components/seed
CLUSTER_SCOPED = {"Namespace", "PriorityClass"}
KUBECTL = shutil.which("kubectl")


def render(overlay: str) -> list[dict[str, Any]]:
    if KUBECTL is None:
        pytest.skip("kubectl not installed")
    out = subprocess.run(
        [KUBECTL, "kustomize", str(K8S / "overlays" / overlay)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [doc for doc in yaml.safe_load_all(out) if doc]


def by_kind(docs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [d for d in docs if d["kind"] == kind]


def pod_specs(docs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    specs = []
    for d in docs:
        if d["kind"] == "Deployment":
            specs.append((d["metadata"]["name"], d["spec"]["template"]["spec"]))
        elif d["kind"] == "CronJob":
            specs.append(
                (d["metadata"]["name"], d["spec"]["jobTemplate"]["spec"]["template"]["spec"])
            )
        elif d["kind"] == "Job":
            specs.append((d["metadata"]["name"], d["spec"]["template"]["spec"]))
    return specs


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_the_api_is_a_single_writer(overlay: str) -> None:
    docs = render(overlay)
    api = next(d for d in by_kind(docs, "Deployment") if d["metadata"]["name"] == "gridcast-api")
    assert api["spec"]["replicas"] == 1
    assert api["spec"]["strategy"]["type"] == "Recreate"
    claims = [
        v["persistentVolumeClaim"]["claimName"]
        for v in api["spec"]["template"]["spec"]["volumes"]
        if "persistentVolumeClaim" in v
    ]
    assert claims == ["gridcast-data"]
    # no other pod mounts the data volume
    for name, spec in pod_specs(docs):
        if name != "gridcast-api":
            assert not any("persistentVolumeClaim" in v for v in spec.get("volumes", [])), name
    assert not by_kind(docs, "HorizontalPodAutoscaler")
    pvc = by_kind(docs, "PersistentVolumeClaim")[0]
    expected = ["ReadWriteOncePod"] if overlay == "gke" else ["ReadWriteOnce"]
    assert pvc["spec"]["accessModes"] == expected


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_the_in_app_scheduler_is_off(overlay: str) -> None:
    config = by_kind(render(overlay), "ConfigMap")
    params = next(c for c in config if c["metadata"]["name"].startswith("gridcast-config"))
    assert params["data"]["SCHEDULER_ENABLED"] == "0"


@pytest.mark.parametrize("overlay", OVERLAYS + SEED_OVERLAYS)
def test_every_container_has_measured_resources_and_runs_hardened(overlay: str) -> None:
    for name, spec in pod_specs(render(overlay)):
        assert spec.get("automountServiceAccountToken") is False, name
        assert spec["securityContext"].get("runAsNonRoot") is True, name
        for c in spec.get("initContainers", []) + spec["containers"]:
            res = c["resources"]
            assert {"cpu", "memory"} <= set(res["requests"]), (name, c["name"])
            assert res["limits"]["memory"] == res["requests"]["memory"], (name, c["name"])
            assert c["securityContext"]["readOnlyRootFilesystem"] is True, (name, c["name"])
            assert c["securityContext"]["allowPrivilegeEscalation"] is False, (name, c["name"])


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_probes_split_liveness_from_readiness(overlay: str) -> None:
    docs = render(overlay)
    api = next(d for d in by_kind(docs, "Deployment") if d["metadata"]["name"] == "gridcast-api")
    c = api["spec"]["template"]["spec"]["containers"][0]
    assert c["livenessProbe"]["httpGet"]["path"] == "/livez"
    assert c["startupProbe"]["httpGet"]["path"] == "/livez"
    assert c["readinessProbe"]["httpGet"]["path"] == "/readyz"


@pytest.mark.parametrize("overlay", OVERLAYS + SEED_OVERLAYS)
def test_images_are_never_floating(overlay: str) -> None:
    for name, spec in pod_specs(render(overlay)):
        for c in spec.get("initContainers", []) + spec["containers"]:
            image = c["image"]
            assert not image.endswith(":latest"), (name, image)
            assert ":" in image.rsplit("/", 1)[-1] or "@sha256:" in image, (name, image)
            if overlay == "gke" and image.startswith("us-east1-docker.pkg.dev/"):
                assert re.search(r"@sha256:[0-9a-f]{64}$", image), (name, image)


def test_no_secret_material_in_the_manifests() -> None:
    patterns = re.compile(
        r"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|\"private_key\"|\"type\":\s*\"service_account\"|"
        r"^kind:\s*Secret\b",
        re.MULTILINE,
    )
    hits = [
        str(p.relative_to(K8S))
        for p in K8S.rglob("*")
        if p.is_file() and patterns.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert hits == []
    for overlay in OVERLAYS + SEED_OVERLAYS if KUBECTL else ():
        assert not by_kind(render(overlay), "Secret")


# ---------------------------------------------------------------------------- CronJobs
@pytest.mark.parametrize("overlay", OVERLAYS)
def test_cronjobs_mirror_the_scheduler_and_never_overlap(overlay: str) -> None:
    from app.scheduler import JOB_BY_NAME

    cronjobs = by_kind(render(overlay), "CronJob")
    seen = set()
    for cj in cronjobs:
        spec = cj["spec"]
        command = spec["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["command"]
        name = command[-1]
        job = JOB_BY_NAME[name]
        seen.add(name)
        day = "1" if job.monthly else "*"
        assert spec["schedule"] == f"{job.at.minute} {job.at.hour} {day} * *", name
        assert spec["timeZone"] == "America/New_York"
        assert spec["concurrencyPolicy"] == "Forbid"
        assert spec["startingDeadlineSeconds"] >= 21600
        assert spec["jobTemplate"]["spec"]["backoffLimit"] >= 3
        assert spec.get("suspend", False) is (overlay == "kind")  # kind never fires on its own
    assert seen == set(JOB_BY_NAME)


def _load_trigger() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("trigger", K8S / "base" / "trigger.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("status", "exit_code"), [(200, 0), (409, 0), (500, 1), (401, 1)])
def test_trigger_exit_codes(status: int, exit_code: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            seen["path"] = self.path
            seen["token"] = self.headers.get("X-Admin-Token", "")
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"status": "x"}')

        def log_message(self, *args: object) -> None:  # keep test output quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("API_BASE", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("ADMIN_TOKEN", "t0ken")
        assert _load_trigger().main(["trigger.py", "ingest_score"]) == exit_code
    finally:
        server.shutdown()
    assert seen == {"path": "/admin/jobs/ingest_score/run", "token": "t0ken"}


def test_trigger_fails_when_the_api_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE", "http://127.0.0.1:9")
    monkeypatch.setenv("TRIGGER_TIMEOUT_SECONDS", "2")
    assert _load_trigger().main(["trigger.py", "forecast_all"]) == 1


# ---------------------------------------------------------------------------- model pin
def _model_config(docs: list[dict[str, Any]]) -> dict[str, str]:
    cm = next(
        c for c in by_kind(docs, "ConfigMap") if c["metadata"]["name"].startswith("gridcast-model")
    )
    return dict(cm["data"])


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_the_model_is_a_pinned_image_copied_by_an_init_container(overlay: str) -> None:
    docs = render(overlay)
    deployments = {d["metadata"]["name"]: d for d in by_kind(docs, "Deployment")}
    api = deployments["gridcast-api"]["spec"]["template"]["spec"]
    init = next(c for c in api["initContainers"] if c["name"] == "model")
    assert "gridcast-model" in init["image"] or init["image"].split("@")[0].endswith("/model")
    models = next(v for v in api["volumes"] if v["name"] == "models")
    assert "emptyDir" in models  # never on the data volume: a rollback restores the model
    pin = _model_config(docs)
    assert pin["REQUIRE_TFT"] == "1" and pin["TFT_EXPECTED_VERSION"].startswith("tft-onnx:")
    assert pin["TFT_BUNDLE_DIR"] == "/models/tft"
    change_cause = deployments["gridcast-api"]["metadata"]["annotations"][
        "kubernetes.io/change-cause"
    ]
    assert pin["TFT_EXPECTED_VERSION"] in change_cause
    # the UI never reads the pin, so a model rollout restarts the API only
    ui = deployments["gridcast-ui"]["spec"]["template"]["spec"]["containers"][0]
    assert all(not e["configMapRef"]["name"].startswith("gridcast-model") for e in ui["envFrom"])


def test_the_kind_pin_is_the_committed_fixture_bundle() -> None:
    from models.tft_onnx import OnnxTFT

    bundle = OnnxTFT.load(K8S / "fixtures" / "tft-tiny")  # raises if FEATURE_VERSION moved on
    assert _model_config(render("kind"))["TFT_EXPECTED_VERSION"] == bundle.version


@pytest.mark.parametrize("overlay", OVERLAYS + SEED_OVERLAYS)
def test_every_object_lives_in_the_gridcast_namespace(overlay: str) -> None:
    # objects an overlay adds don't inherit the base's namespace: this caught the archive
    # fixture landing in `default`, where the API could not resolve it
    outside = [
        (d["kind"], d["metadata"]["name"])
        for d in render(overlay)
        if d["kind"] not in CLUSTER_SCOPED and d["metadata"].get("namespace") != "gridcast"
    ]
    assert outside == []


# ---------------------------------------------------------------------------- seed
def _object_id(d: dict[str, Any]) -> tuple[str, str]:
    return d["kind"], d["metadata"]["name"]


def _api(docs: list[dict[str, Any]]) -> dict[str, Any]:
    return next(d for d in by_kind(docs, "Deployment") if d["metadata"]["name"] == "gridcast-api")


@pytest.mark.parametrize("overlay", SEED_OVERLAYS)
def test_the_seed_is_the_only_writer_while_it_runs(overlay: str) -> None:
    docs = render(overlay)
    assert _api(docs)["spec"]["replicas"] == 0
    assert all(cj["spec"]["suspend"] is True for cj in by_kind(docs, "CronJob"))
    (job,) = by_kind(docs, "Job")
    pod = job["spec"]["template"]["spec"]
    api = _api(docs)["spec"]["template"]["spec"]
    claims = [
        v["persistentVolumeClaim"]["claimName"]
        for v in pod["volumes"]
        if "persistentVolumeClaim" in v
    ]
    assert claims == ["gridcast-data"]
    # the seed runs the API's own image, model and settings
    assert pod["containers"][0]["image"] == api["containers"][0]["image"]
    assert pod["initContainers"][0]["image"] == api["initContainers"][0]["image"]
    assert pod["containers"][0]["envFrom"] == api["containers"][0]["envFrom"]
    assert job["spec"]["backoffLimit"] <= 1 and job["spec"]["activeDeadlineSeconds"] > 0


@pytest.mark.parametrize("overlay", SEED_OVERLAYS)
def test_a_seed_overlay_is_its_environment_plus_the_seed_only(overlay: str) -> None:
    # re-applying the plain environment after the seed must restore it exactly
    seeded = [d for d in render(overlay) if d["kind"] != "Job"]
    for d in seeded:
        if d["kind"] == "CronJob":
            del d["spec"]["suspend"]
        elif d["kind"] == "Deployment" and d["metadata"]["name"] == "gridcast-api":
            d["spec"]["replicas"] = 1
    plain = render(overlay.removesuffix("-seed"))
    for d in plain:
        if d["kind"] == "CronJob":
            d["spec"].pop("suspend", None)
    assert sorted(seeded, key=_object_id) == sorted(plain, key=_object_id)


@pytest.mark.parametrize("overlay", OVERLAYS + SEED_OVERLAYS)
def test_the_writer_outranks_every_other_gridcast_pod(overlay: str) -> None:
    # on a full node a system pod evicts the lowest priority first: never the single writer
    docs = render(overlay)
    (pc,) = by_kind(docs, "PriorityClass")
    assert 0 < pc["value"] < 1_000_000_000 and pc["globalDefault"] is False
    writers = {"gridcast-api", "gridcast-seed"}
    for name, spec in pod_specs(docs):
        expected = pc["metadata"]["name"] if name in writers else None
        assert spec.get("priorityClassName") == expected, name
