# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(90),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_DIR = _REPO_ROOT / "examples/backends/vllm/deploy/kvcr"
_MANIFESTS = ("agg.yaml", "agg-memory-service.yaml")
_GUARD_DAEMONSET_WORKER = "agg-guard-daemonset.yaml"
_STATE_AGENT = "agg-state-agent.yaml"
_MODEL_NAMESPACE = "kvcr-qwen3-0-6b"


def _worker(manifest_name: str) -> dict:
    manifest = yaml.safe_load((_MANIFEST_DIR / manifest_name).read_text())
    return next(
        component
        for component in manifest["spec"]["components"]
        if component["name"] == "worker"
    )


def _kv_transfer_config_python(command: str) -> str:
    marker = "python3 - <<'PY'\n"
    start = command.index(marker) + len(marker)
    end = command.index("\nPY\n", start)
    return command[start:end]


@pytest.mark.parametrize("manifest_name", _MANIFESTS)
def test_kvcr_variants_require_two_gpu_rdma_nodes(manifest_name: str) -> None:
    worker = _worker(manifest_name)
    pod_spec = worker["podTemplate"]["spec"]
    main = pod_spec["containers"][0]

    assert worker["replicas"] == 2
    anti_affinity = pod_spec["affinity"]["podAntiAffinity"]
    assert anti_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
    assert "preferredDuringSchedulingIgnoredDuringExecution" not in anti_affinity

    for resource_class in ("requests", "limits"):
        resources = main["resources"][resource_class]
        assert resources["nvidia.com/gpu"] == "1"
        assert resources["${DYNAMO_RDMA_RESOURCE}"] == "1"

    env = {item["name"]: item.get("value") for item in main["env"]}
    assert env["UCX_TLS"] == "rc_x,cuda"
    assert env["UCX_NET_DEVICES"] == "${DYNAMO_UCX_NET_DEVICES}"
    assert all(
        item["name"] != "UCX_PROTO_INFO"
        for container in pod_spec["containers"]
        for item in container["env"]
    )
    assert env["KVCR_CACHE_SLOT_COUNT"] == "2"
    slot = next(item for item in main["env"] if item["name"] == "KVCR_CACHE_SLOT")
    assert slot["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['grove.io/podclique-pod-index']"
    )
    assert "IPC_LOCK" in main["securityContext"]["capabilities"]["add"]


@pytest.mark.parametrize("manifest_name", _MANIFESTS)
def test_kvcr_cache_slot_is_validated_and_formatted(manifest_name: str) -> None:
    main = _worker(manifest_name)["podTemplate"]["spec"]["containers"][0]
    script = _kv_transfer_config_python(main["args"][0])
    env = {
        **os.environ,
        "KVCR_CACHE_SLOT": "1",
        "KVCR_CACHE_SLOT_COUNT": "2",
        "POD_IP": "192.0.2.1",
    }

    result = subprocess.run(
        ["python3", "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    config = json.loads(result.stdout)
    owner = config["kv_connector_extra_config"]["dynamo_state_agent"][
        "cache_owner_ids"
    ]["0"]
    assert owner.endswith("/00000000000000000000000000000001")

    env["KVCR_CACHE_SLOT"] = "2"
    result = subprocess.run(
        ["python3", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "outside [0, 2)" in result.stderr


def test_process_local_variant_couples_state_agent_and_vllm() -> None:
    manifest = yaml.safe_load((_MANIFEST_DIR / "agg.yaml").read_text())
    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-discovery-backend"]
        == "etcd"
    )
    pod_spec = _worker("agg.yaml")["podTemplate"]["spec"]
    main = pod_spec["containers"][0]
    command = main["args"][0]

    assert len(pod_spec["containers"]) == 1
    assert "initContainers" not in pod_spec
    assert "python3 -m dynamo.kv_state_agent" in command
    assert "python3 -m dynamo.vllm" in command
    assert 'if [ "$POD_INDEX" = "0" ]' in command
    assert "env -u DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS" in command
    assert "DYN_DISCOVERY_BACKEND=etcd" in command
    assert "DYN_SYSTEM_PORT=9091" in command
    assert "DYN_HEALTH_CHECK_ENABLED=false" in command
    assert '--max-slots "$KVCR_CACHE_SLOT_COUNT"' in command
    assert 'wait -n "$state_agent_pid" "$vllm_pid"' in command
    assert "kvcr.kvcr_service" not in command
    assert 'case "$POD_INDEX" in' not in command
    assert '"cache_owner_ids": {"0": cache_owner_id()}' in command
    assert all(item["name"] != "POD_UID" for item in main["env"])

    assert "startupProbe" not in main
    assert "livenessProbe" not in main
    assert "readinessProbe" not in main


def test_memory_service_variant_keeps_guard_in_sidecar() -> None:
    manifest = yaml.safe_load((_MANIFEST_DIR / "agg-memory-service.yaml").read_text())
    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-kube-discovery-mode"]
        == "container"
    )
    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-discovery-backend"]
        == "kubernetes"
    )
    pod_spec = _worker("agg-memory-service.yaml")["podTemplate"]["spec"]
    main, sidecar = pod_spec["containers"]
    main_command = main["args"][0]
    sidecar_command = sidecar["args"][0]

    assert sidecar["name"] == "kvcr-services"
    assert "python3 -m kvcr.kvcr_service" in sidecar_command
    assert "python3 -m dynamo.kv_state_agent" in sidecar_command
    assert "--guard-count 1" in sidecar_command
    assert "--pool-sizes-gb 2" in sidecar_command
    assert 'if [ "$POD_INDEX" = "0" ]' in sidecar_command
    assert "DYN_HEALTH_CHECK_ENABLED=false" in sidecar_command
    assert "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS" not in sidecar_command
    assert all(
        item["name"] != "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"
        for item in sidecar["env"]
    )
    assert '--max-slots "$KVCR_CACHE_SLOT_COUNT"' in sidecar_command
    assert 'wait -n "$kvcr_pid" "$state_agent_pid"' in sidecar_command
    assert '"kvcr_service_socket_path": "/run/kvcr/memory.sock"' in main_command
    assert "/run/kvcr/hold-engine-start" not in main_command
    assert 'case "$POD_INDEX" in' not in main_command
    assert '"cache_owner_ids": {"0": cache_owner_id()}' in main_command
    assert all(item["name"] != "POD_UID" for item in main["env"])
    pod_uid = next(item for item in sidecar["env"] if item["name"] == "POD_UID")
    assert pod_uid["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"

    for resource_class in ("requests", "limits"):
        assert sidecar["resources"][resource_class]["${DYNAMO_RDMA_RESOURCE}"] == "1"
    assert "IPC_LOCK" in sidecar["securityContext"]["capabilities"]["add"]

    for container in (main, sidecar):
        mounts = {mount["name"] for mount in container["volumeMounts"]}
        assert mounts == {"kvcr-memory", "kvcr-socket"}
    memory = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "kvcr-memory"
    )
    assert memory["emptyDir"]["medium"] == "Memory"

    for probe_name in ("startupProbe", "livenessProbe", "readinessProbe"):
        probe = sidecar[probe_name]["exec"]["command"][-1]
        assert "/run/kvcr/memory.sock" in probe
        assert "9091/live" in probe
        assert "os.environ['POD_INDEX'] != '0' or" in probe


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="envsubst is unavailable")
@pytest.mark.parametrize("memory_service", ("false", "true"))
def test_deploy_script_renders_selected_variant(memory_service: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_RDMA_RESOURCE": "rdma/test",
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "qwen3-0.6b-example-v1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": memory_service,
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-only"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )

    rendered = yaml.safe_load(result.stdout)
    assert rendered["kind"] == "DynamoGraphDeployment"
    containers = [
        container
        for component in rendered["spec"]["components"]
        for container in component["podTemplate"]["spec"]["containers"]
    ]
    assert all(
        container["image"] == "runtime:1.4.0@sha256:test" for container in containers
    )
    worker = next(
        component
        for component in rendered["spec"]["components"]
        if component["name"] == "worker"
    )
    for container in worker["podTemplate"]["spec"]["containers"]:
        env_by_name = {item["name"]: item.get("value") for item in container["env"]}
        assert env_by_name["UCX_NET_DEVICES"] == "mlx5_test:1"
        assert container["resources"]["limits"]["rdma/test"] == "1"
    if memory_service == "true":
        assert result.stdout.count("qwen3-0.6b-example-v1") == 2
    expected_name = (
        "vllm-agg-kvcr-memory-service" if memory_service == "true" else "vllm-agg-kvcr"
    )
    assert f"name: {expected_name}" in result.stdout


def test_deploy_script_rejects_unknown_argument() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": "false",
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-onyl"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_deploy_script_rejects_unsafe_compatibility_digest() -> None:
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_KVCR_COMPATIBILITY_DIGEST": "unsafe value",
            "DYNAMO_UCX_NET_DEVICES": "mlx5_test:1",
            "DYNAMO_VLLM_IMAGE": "runtime:1.4.0@sha256:test",
            "KVCR_MEMORY_SERVICE_ENABLED": "true",
        }
    )

    result = subprocess.run(
        [str(_MANIFEST_DIR / "deploy.sh"), "--render-only"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unsupported character" in result.stderr


def _manifest_text(name: str) -> str:
    return (_MANIFEST_DIR / name).read_text()


def _expected_slot(node_name: str, guard_index: int) -> str:
    digest = hashlib.sha256(
        "\0".join(["kvcr-guard-slot/v1", node_name, str(guard_index)]).encode()
    ).digest()
    return digest[:16].hex()


def _guard_owner(node_name: str, guard_index: str = "0", count: str = "1", **extra):
    main = _worker(_GUARD_DAEMONSET_WORKER)["podTemplate"]["spec"]["containers"][0]
    script = _kv_transfer_config_python(main["args"][0])
    env = {
        **os.environ,
        "KVCR_NODE_NAME": node_name,
        "KVCR_GUARD_INDEX": guard_index,
        "KVCR_GUARD_COUNT": count,
        "POD_IP": "192.0.2.1",
        **extra,
    }
    return subprocess.run(
        ["python3", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )


def test_guard_daemonset_slot_is_derived_from_node_and_guard_index() -> None:
    def owner(**kwargs) -> str:
        result = _guard_owner(**kwargs)
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        return config["kv_connector_extra_config"]["dynamo_state_agent"][
            "cache_owner_ids"
        ]["0"]

    node_a = owner(node_name="node-a")
    slot = node_a.rsplit("/", 1)[1]
    assert len(slot) == 32
    assert slot == _expected_slot("node-a", 0)
    # Node-stable: pod identity and pod index must not influence the slot.
    assert owner(node_name="node-a", POD_INDEX="1", POD_NAME="other") == node_a
    assert owner(node_name="node-b") != node_a
    assert owner(node_name="node-a", guard_index="1", count="2").rsplit("/", 1)[
        1
    ] == _expected_slot("node-a", 1)

    for kwargs, message in (
        ({"node_name": ""}, "node name must not be empty"),
        ({"node_name": "n", "guard_index": "x"}, "must be decimal integers"),
        ({"node_name": "n", "guard_index": "1", "count": "1"}, "outside [0, 1)"),
    ):
        result = _guard_owner(**kwargs)
        assert result.returncode != 0
        assert message in result.stderr


def test_guard_daemonset_variant_uses_external_state_agent() -> None:
    manifest = yaml.safe_load((_MANIFEST_DIR / _GUARD_DAEMONSET_WORKER).read_text())
    pod_spec = _worker(_GUARD_DAEMONSET_WORKER)["podTemplate"]["spec"]
    main = pod_spec["containers"][0]
    command = main["args"][0]

    assert (
        manifest["metadata"]["annotations"]["nvidia.com/dynamo-discovery-backend"]
        == "kubernetes"
    )
    # No sidecar and no in-pod state agent or Guard.
    assert [container["name"] for container in pod_spec["containers"]] == ["main"]
    assert "dynamo.kv_state_agent" not in _manifest_text(_GUARD_DAEMONSET_WORKER)
    assert "kvcr.kvcr_service" not in _manifest_text(_GUARD_DAEMONSET_WORKER)
    assert f"--kv-state-endpoint {_MODEL_NAMESPACE}/backend/generate" in command

    # The worker waits for the Guard socket instead of crash-looping.
    (wait,) = pod_spec["initContainers"]
    assert wait["name"] == "wait-for-guard"
    assert "/run/kvcr/qwen3-0.6b/memory.sock" in wait["args"][0]
    assert "/run/kvcr/qwen3-0.6b/memory.sock" in command

    # The owner slot comes from the node, not the pod index.
    env = {item["name"]: item for item in main["env"]}
    assert env["KVCR_NODE_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "spec.nodeName"
    )
    assert "KVCR_CACHE_SLOT" not in env
    assert "POD_INDEX" not in env
    assert "podclique-pod-index" not in _manifest_text(_GUARD_DAEMONSET_WORKER)
    assert "kvcr-guard-slot/v1" in command
    for volume in pod_spec["volumes"]:
        assert "hostPath" in volume
    assert "hostPID" not in pod_spec
    assert "hostNetwork" not in pod_spec


def test_state_agent_deployment_is_singleton_in_the_model_namespace() -> None:
    documents = list(yaml.safe_load_all(_manifest_text(_STATE_AGENT)))
    deployment = next(doc for doc in documents if doc["kind"] == "Deployment")
    spec = deployment["spec"]
    pod_spec = spec["template"]["spec"]
    (container,) = pod_spec["containers"]
    env = {item["name"]: item for item in container["env"]}

    assert spec["replicas"] == 1
    assert spec["strategy"]["type"] == "Recreate"
    assert env["DYN_NAMESPACE"]["value"] == _MODEL_NAMESPACE
    assert env["DYN_DISCOVERY_BACKEND"]["value"] == "kubernetes"
    assert container["command"] + container["args"][:2] == [
        "python3",
        "-m",
        "dynamo.kv_state_agent",
    ]
    assert "--max-slots" in container["args"]
    for name, field in (
        ("POD_NAME", "metadata.name"),
        ("POD_NAMESPACE", "metadata.namespace"),
        ("POD_UID", "metadata.uid"),
    ):
        assert env[name]["valueFrom"]["fieldRef"]["fieldPath"] == field
    labels = spec["template"]["metadata"]["labels"]
    assert labels["nvidia.com/dynamo-discovery-backend"] == "kubernetes"
    assert labels["nvidia.com/dynamo-discovery-enabled"] == "true"
    assert not pod_spec.get("hostPID") and not pod_spec.get("hostNetwork")

    role = next(doc for doc in documents if doc["kind"] == "Role")
    assert any("dynamoworkermetadatas" in rule["resources"] for rule in role["rules"])
    assert pod_spec["serviceAccountName"] == next(
        doc["metadata"]["name"] for doc in documents if doc["kind"] == "ServiceAccount"
    )
    # Workers target the same namespace the agent registers in.
    command = _worker(_GUARD_DAEMONSET_WORKER)["podTemplate"]["spec"]["containers"][0][
        "args"
    ][0]
    assert f"--kv-state-endpoint {env['DYN_NAMESPACE']['value']}/backend/generate" in (
        command
    )
