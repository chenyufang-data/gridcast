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


@pytest.mark.parametrize("overlay", OVERLAYS)
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


@pytest.mark.parametrize("overlay", OVERLAYS)
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
    for overlay in OVERLAYS if KUBECTL else ():
        assert not by_kind(render(overlay), "Secret")
