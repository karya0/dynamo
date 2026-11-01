# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import stat
import uuid
from pathlib import Path

import jsonschema
import pytest
import yaml

from dynamo.gaas import (
    claim_spec,
    launch_argv,
    main,
    prepare_claim,
    validate_descriptor,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.fixture
def claim():
    return claim_spec("pool-a", 1.0, "model-layout-digest")


@pytest.fixture
def broker_socket():
    # A logical endpoint only: these tests never open a network or UNIX socket.
    return "/node-shm/gaas-test/broker.sock"


@pytest.fixture
def descriptor(claim, broker_socket):
    return {
        **claim,
        "state": "ready",
        "holder_uid": str(uuid.uuid4()),
        "holder_pod_ip": "192.0.2.15",
        "service_socket_path": str(Path(broker_socket).parent / "p-a" / "service.sock"),
        "control_port": 55001,
        "guard_index": 0,
    }


@pytest.mark.parametrize("pool_id", ["../pool", "Upper", "", "a" * 64])
def test_reject_bad_pool_id(pool_id):
    with pytest.raises(ValueError, match="DNS label"):
        claim_spec(pool_id, 1.0, "digest")


@pytest.mark.parametrize("size", [True, 0, 2, float("nan")])
def test_reject_unsupported_capacity(size):
    with pytest.raises(ValueError, match="size_gb"):
        claim_spec("pool", size, "digest")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pool_id", "another"),
        ("size_gb", 0.5),
        ("size_gb", True),
        ("compatibility_digest", "another"),
        ("holder_uid", "not-a-uid"),
        ("holder_pod_ip", "holder-name"),
        ("holder_pod_ip", "0.0.0.0"),
        ("holder_pod_ip", "127.0.0.1"),
        ("state", "pending"),
        ("control_port", 0),
        ("guard_index", 1),
        ("service_socket_path", "/outside/service.sock"),
    ],
)
def test_descriptor_fails_closed(claim, descriptor, broker_socket, key, value):
    descriptor[key] = value
    with pytest.raises(ValueError):
        validate_descriptor(descriptor, claim, broker_socket)


def test_init_writes_atomic_private_descriptor(
    tmp_path, claim, descriptor, broker_socket
):
    output = tmp_path / "claim" / "descriptor.json"
    calls = []

    def ensure(socket_path, request, *, timeout_s):
        calls.append((socket_path, request, timeout_s))
        return descriptor

    prepare_claim(broker_socket, claim, output, 12, ensure)
    assert calls == [(broker_socket, claim, 12)]
    assert json.loads(output.read_text()) == descriptor
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert list(output.parent.iterdir()) == [output]


def test_rejected_init_does_not_replace_old_file(
    tmp_path, claim, descriptor, broker_socket
):
    output = tmp_path / "claim.json"
    output.write_text("previous valid descriptor")
    descriptor["state"] = "pending"
    with pytest.raises(ValueError, match="not ready"):
        prepare_claim(broker_socket, claim, output, 12, lambda *a, **k: descriptor)
    assert output.read_text() == "previous valid descriptor"


def test_launch_uses_pinned_offloading_schema_and_holder_address(descriptor):
    args = launch_argv(descriptor, 64 * 1024**2, ["--model", "model with space"])
    assert args[1:3] == ["-m", "dynamo.vllm"]
    assert args[3:5] == ["--model", "model with space"]
    config = json.loads(args[-1])
    assert args[-2] == "--kv-transfer-config"
    assert config["kv_connector"] == "OffloadingConnector"
    assert config["kv_role"] == "kv_both"
    extra = config["kv_connector_extra_config"]
    assert extra["spec_name"] == "TieringOffloadingSpec"
    assert extra["cpu_bytes_to_use"] == 64 * 1024**2
    assert extra["self_describing_kv_events"] is True
    tier = extra["secondary_tiers"][0]
    assert tier["control_advertise_host"] == descriptor["holder_pod_ip"]
    assert tier["control_host"] == "0.0.0.0"
    assert tier["control_ports"] == [55001]
    assert tier["kvcr_service_socket_path"] == descriptor["service_socket_path"]
    assert tier["compatibility_digest"] == descriptor["compatibility_digest"]
    assert tier["router_capabilities"] == ["router_hint"]


def test_merge_preserves_user_settings(descriptor):
    old = {
        "engine_id": "engine-a",
        "kv_connector_extra_config": {
            "eviction_policy": "lru",
            "dynamo_state_agent": {"cache_owner_ids": {"0": "owner"}},
            "secondary_tiers": [{"type": "kvcr", "enable_telemetry": True}],
        },
    }
    args = launch_argv(descriptor, 4096, ["--kv-transfer-config=" + json.dumps(old)])
    new = json.loads(args[-1])
    assert new["engine_id"] == "engine-a"
    assert new["kv_connector_extra_config"]["dynamo_state_agent"] == {
        "cache_owner_ids": {"0": "owner"}
    }
    assert new["kv_connector_extra_config"]["secondary_tiers"][0]["enable_telemetry"]


@pytest.mark.parametrize(
    "old",
    [
        {"kv_connector": "OtherConnector"},
        {"kv_role": "kv_producer"},
        {"kv_connector_extra_config": {"spec_name": "CPUOffloadingSpec"}},
        {"kv_connector_extra_config": {"cpu_bytes_to_use": 123}},
        {"kv_connector_extra_config": {"secondary_tiers": [{"type": "fs"}]}},
        {
            "kv_connector_extra_config": {
                "secondary_tiers": [{"type": "kvcr", "control_ports": [123]}]
            }
        },
    ],
)
def test_merge_rejects_conflicts(descriptor, old):
    with pytest.raises(ValueError):
        launch_argv(descriptor, 4096, ["--kv-transfer-config", json.dumps(old)])


@pytest.mark.parametrize(
    "args",
    [
        ["--tensor-parallel-size", "2"],
        ["--pipeline-parallel-size=2"],
        ["-dp", "2"],
    ],
)
def test_reject_multi_worker_contract(descriptor, args):
    with pytest.raises(ValueError, match="TP=PP=DP=1"):
        launch_argv(descriptor, 4096, args)


def test_launch_requires_gate_file(tmp_path, broker_socket):
    with pytest.raises(FileNotFoundError):
        main(
            [
                "launch",
                "--broker-socket",
                broker_socket,
                "--claim-file",
                str(tmp_path / "missing.json"),
                "--pool-id",
                "pool-a",
                "--size-gb",
                "1",
                "--compatibility-digest",
                "model-layout-digest",
                "--cpu-bytes",
                "4096",
                "--",
                "--model",
                "model",
            ],
            ensure_pool=lambda *a, **k: pytest.fail(
                "launch must not ensure a new pool"
            ),
        )


def test_main_execs_without_shell(tmp_path, descriptor, broker_socket, monkeypatch):
    claim_file = tmp_path / "claim.json"
    claim_file.write_text(json.dumps(descriptor))
    calls = []
    monkeypatch.setattr(
        os, "execv", lambda executable, args: calls.append((executable, args))
    )
    assert (
        main(
            [
                "launch",
                "--broker-socket",
                broker_socket,
                "--claim-file",
                str(claim_file),
                "--pool-id",
                "pool-a",
                "--size-gb",
                "1",
                "--compatibility-digest",
                "model-layout-digest",
                "--cpu-bytes",
                "4096",
                "--",
                "--model",
                "model with space",
            ],
            ensure_pool=lambda *a, **k: pytest.fail("unexpected broker call"),
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0][1][3:5] == ["--model", "model with space"]


def test_canonical_templates_match_yaml_and_single_worker_contract():
    root = Path(__file__).parents[5]
    examples = root / "examples/backends/vllm/gaas"
    bundle = json.loads((examples / "manifest-bundle.json").read_text())
    yaml_resources = list(
        yaml.safe_load_all((examples / "guard-daemonset.yaml").read_text())
    )
    yaml_resources.append(yaml.safe_load((examples / "dgd.yaml").read_text()))
    assert bundle["resources"] == yaml_resources
    assert [item["kind"] for item in yaml_resources] == [
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "DaemonSet",
        "DynamoGraphDeployment",
    ]
    role = yaml_resources[1]
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["pods"],
            "verbs": ["get", "list", "watch", "create", "delete"],
        }
    ]
    guard = yaml_resources[3]["spec"]["template"]["spec"]
    assert guard["nodeSelector"] == {"kubernetes.io/hostname": "${GAAS_NODE}"}
    assert guard["containers"][0]["env"][-2:] == [
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
        {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
    ]
    worker = yaml_resources[4]["spec"]["services"]["VllmWorker"]["extraPodSpec"]
    assert worker["mainContainer"]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "nvidia.com/gpu" not in worker["initContainers"][0]["resources"]["limits"]
    assert all("hostNetwork" not in pod for pod in (guard, worker))
    assert worker["mainContainer"]["volumeMounts"][-1]["readOnly"] is True


def test_rendered_dgd_matches_repository_crd_schema():
    root = Path(__file__).parents[5]
    source = (root / "examples/backends/vllm/gaas/manifest-bundle.json").read_text()
    variables = {
        "GAAS_NAME": "gaas-test",
        "GAAS_NAMESPACE": "gaas-example",
        "GAAS_NODE": "node-a",
        "GAAS_SOCKET_ROOT": "/node-shm/g7-test",
        "GAAS_IMAGE": "registry.example/image@sha256:" + "a" * 64,
        "GAAS_COMPATIBILITY_DIGEST": "c" * 64,
        "GAAS_MODEL": "org/model",
        "GAAS_MODEL_REVISION": "b" * 40,
        "GAAS_CPU_PRIMARY_BYTES": "67108864",
        "GAAS_GPU_KV_BYTES": "268435456",
        "GAAS_MAX_MODEL_LEN": "8192",
    }
    rendered = re.sub(r"\$\{(GAAS_[A-Z_]+)\}", lambda m: variables[m[1]], source)
    dgd = json.loads(rendered)["resources"][-1]
    crd = yaml.safe_load(
        (
            root
            / "deploy/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml"
        ).read_text()
    )
    version = next(v for v in crd["spec"]["versions"] if v["name"] == "v1alpha1")
    jsonschema.Draft4Validator(version["schema"]["openAPIV3Schema"]).validate(dgd)
