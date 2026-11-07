# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import hashlib
import json
import logging
import mmap
import os
import re
import stat
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from collections import defaultdict
from typing import Any

import jsonschema
import pytest
import yaml
from kvcr.service.protocol import (
    KVCRServiceReasonError,
    PoolRegion,
    _Grown,
    _Shrunk,
)

from dynamo.gaas import (
    claim_spec,
    launch_argv,
    main,
    prepare_claim,
    validate_descriptor,
    validate_gaas_descriptor,
    wait_for_socket,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.fixture(autouse=True)
def no_external_commands(tmp_path, monkeypatch):
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("")
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))

    def unexpected(*args, **kwargs):
        pytest.fail("unexpected external command")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(os, "execv", unexpected)


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


@pytest.mark.parametrize("size", [0.5, 0.5 + 1 / 1024, 1, 2, 12, 150, 256, "256"])
def test_claim_capacity_roundtrips(size):
    claim = claim_spec("pool", size, "digest")
    assert json.loads(json.dumps(claim)) == {
        "pool_id": "pool",
        "size_gb": float(size),
        "compatibility_digest": "digest",
    }


@pytest.mark.parametrize(
    "size",
    [
        True,
        False,
        0,
        -1,
        0.5 - 1 / 1024,
        256 + 1 / 1024,
        1 + 1 / 2048,
        float("nan"),
        float("inf"),
        -float("inf"),
        "invalid",
        None,
    ],
)
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


@pytest.mark.parametrize("size_gb", [1.0, 12.0, 150.0, 256.0])
def test_init_writes_atomic_private_descriptor(
    tmp_path, claim, descriptor, broker_socket, size_gb
):
    claim["size_gb"] = descriptor["size_gb"] = size_gb
    output = tmp_path / "claim" / "descriptor.json"
    calls = []

    def ensure(socket_path, request, *, timeout_s):
        calls.append((socket_path, request, timeout_s))
        return descriptor

    assert (
        main(
            [
                "init",
                "--broker-socket",
                broker_socket,
                "--claim-file",
                str(output),
                "--pool-id",
                claim["pool_id"],
                "--size-gb",
                str(size_gb),
                "--compatibility-digest",
                claim["compatibility_digest"],
                "--timeout-s",
                "12",
            ],
            ensure_pool=ensure,
        )
        == 0
    )
    assert calls == [(broker_socket, claim, 12)]
    assert json.loads(output.read_text()) == descriptor
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("command", ["init", "launch"])
@pytest.mark.parametrize("size_gb", ["256.00000000000001", "150.00000000000001"])
def test_cli_rejects_capacity_before_float_rounding(
    tmp_path, broker_socket, command, size_gb
):
    output = tmp_path / "claim.json"
    args = [
        command,
        "--broker-socket",
        broker_socket,
        "--claim-file",
        str(output),
        "--pool-id",
        "pool",
        "--size-gb",
        size_gb,
        "--compatibility-digest",
        "digest",
    ]
    if command == "launch":
        args.extend(["--cpu-bytes", "64"])

    def unexpected(*args, **kwargs):
        pytest.fail("invalid capacity reached the holder")

    with pytest.raises(ValueError, match="whole MiB"):
        main(args, ensure_pool=unexpected)
    assert not output.exists()


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
        {"kv_connector_module_path": "other_connector"},
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
        ["--tensor_parallel_size", "2"],
        ["--tensor_parallel_size=2"],
        ["--pipeline-parallel-size=2"],
        ["--pipeline_parallel_size=2"],
        ["--data_parallel_size", "2"],
        ["-pp=2"],
        ["-dp", "2"],
    ],
)
def test_reject_multi_worker_contract(descriptor, args):
    with pytest.raises(ValueError, match="TP=PP=DP=1"):
        launch_argv(descriptor, 4096, args)


@pytest.mark.parametrize(
    "args",
    [
        ["--tensor-parallel-s", "2"],
        ["--tensor_parallel_s=2"],
        ["--pipeline-parallel-s=2"],
        ["--data-parallel-s=2"],
        ["-t", "2"],
        ["--kv-transfer-conf", "{}"],
        ["--kv-transfer-config.kv_connector", '"OtherConnector"'],
        ["--kv_transfer_config.kv_connector=OtherConnector"],
        ["--kv-transfer-c.kv_connector=OtherConnector"],
        ["--kv-transfer-config.kv_connector_extra_config.spec_name=CPUOffloadingSpec"],
        ["--config", "backend.yaml"],
        ["--config=backend.yaml"],
        ["--", "--tensor-parallel-size", "2"],
    ],
)
def test_reject_unvalidated_native_argument_styles(descriptor, args):
    with pytest.raises(ValueError, match="GaaS"):
        launch_argv(descriptor, 4096, args)


def test_normalize_guarded_names_without_changing_values(descriptor):
    args = launch_argv(
        descriptor,
        4096,
        [
            "--model=org/model_with_underscores",
            "--tensor_parallel_size=1",
            "--kv_transfer_config",
            '{"engine_id":"engine_with_underscores"}',
            "--compilation_config.cudagraph_mode",
            '"piecewise"',
        ],
    )
    assert args[3:-2] == [
        "--model=org/model_with_underscores",
        "--tensor-parallel-size=1",
        "--compilation_config.cudagraph_mode",
        '"piecewise"',
    ]
    assert json.loads(args[-1])["engine_id"] == "engine_with_underscores"
    assert args.count("--kv-transfer-config") == 1


@pytest.fixture(scope="module")
def pinned_native_parser():
    """Exercise the parser from vLLM 5df63a7 without importing an engine."""
    source_path = os.environ.get("GAAS_VLLM_ARGPARSE_SOURCE")
    if source_path is None:
        pytest.skip("set GAAS_VLLM_ARGPARSE_SOURCE to the pinned argparse_utils.py")
    source = Path(source_path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "d1330ebc9e8fc4ef0ff03a985b8bbf2551ca10780d427a830465a97253558a78"
    )
    tree = ast.parse(source)
    parser_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlexibleArgumentParser"
    )
    parser_class.body = [
        node
        for node in parser_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    ]
    integer_parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "human_readable_int"
    )
    namespace = {
        "argparse": argparse,
        "ArgumentParser": argparse.ArgumentParser,
        "ArgumentTypeError": argparse.ArgumentTypeError,
        "Namespace": argparse.Namespace,
        "Any": Any,
        "json": json,
        "re": pytest.importorskip("regex"),
        "sys": __import__("sys"),
        "defaultdict": defaultdict,
        "logger": logging.getLogger(__name__),
    }
    module = ast.Module(body=[integer_parser, parser_class], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)

    def make_parser():
        parser = namespace["FlexibleArgumentParser"]()
        for name, short in (
            ("tensor-parallel-size", "-tp"),
            ("pipeline-parallel-size", "-pp"),
            ("data-parallel-size", "-dp"),
        ):
            parser.add_argument(f"--{name}", short, type=int, default=1)
        parser.add_argument("--kv-transfer-config", type=json.loads)
        parser.add_argument("--model")
        parser.add_argument("--compilation-config", type=json.loads)
        return parser

    return make_parser


def test_pinned_native_parser_reproduces_reported_bypasses(pinned_native_parser):
    assert (
        pinned_native_parser()
        .parse_args(["--tensor_parallel_size", "2"])
        .tensor_parallel_size
        == 2
    )
    parsed = pinned_native_parser().parse_args(
        [
            "--kv-transfer-config",
            '{"kv_connector":"OffloadingConnector"}',
            "--kv-transfer-config.kv_connector",
            '"OtherConnector"',
        ]
    )
    assert parsed.kv_transfer_config["kv_connector"] == "OtherConnector"


@pytest.mark.parametrize(
    "topology", [["--tensor_parallel_size=1"], ["-tp", "1"], ["-pp=1"]]
)
def test_pinned_native_parser_receives_one_validated_claim_config(
    descriptor, pinned_native_parser, topology
):
    args = launch_argv(
        descriptor,
        4096,
        [*topology, "--kv_transfer_config", '{"engine_id":"engine_with_underscores"}'],
    )
    native = pinned_native_parser().parse_args(args[3:])
    assert native.tensor_parallel_size == native.pipeline_parallel_size == 1
    assert native.data_parallel_size == 1
    config = native.kv_transfer_config
    assert config["kv_connector"] == "OffloadingConnector"
    assert config["engine_id"] == "engine_with_underscores"
    tier = config["kv_connector_extra_config"]["secondary_tiers"][0]
    assert tier["compatibility_digest"] == descriptor["compatibility_digest"]
    assert tier["control_advertise_host"] == descriptor["holder_pod_ip"]


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


@pytest.mark.parametrize("size_gb", [1.0, 12.0])
def test_main_execs_without_shell(
    tmp_path, descriptor, broker_socket, monkeypatch, size_gb
):
    descriptor["size_gb"] = size_gb
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
                str(size_gb),
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


# -- GaaS (v2) claim: tenant and lineage select it; the supervisor creates the holder.

_GIB = 1 << 30
_V2_FLAGS = [
    "--tenant",
    "t1",
    "--lineage",
    "dgd-a",
    "--minimum-gb",
    "0.5",
    "--maximum-gb",
    "2",
    "--state-agent-namespace",
    "ns",
    "--state-agent-component",
    "backend",
]


@pytest.fixture
def service_socket():
    with tempfile.TemporaryDirectory(prefix="gaas-", dir="/tmp") as directory:
        yield str(Path(directory) / "service.sock")


@pytest.fixture
def v2_claim(service_socket):
    return claim_spec(
        "pool-a",
        1.0,
        "model-layout-digest",
        tenant="t1",
        lineage="dgd-a",
        minimum_gb=0.5,
        maximum_gb=2,
        state_agent_namespace="ns",
        state_agent_component="backend",
        service_socket=service_socket,
    )


def test_v1_claim_spec_is_unchanged_without_gaas_flags():
    assert claim_spec("pool", 1.0, "d") == {
        "pool_id": "pool",
        "size_gb": 1.0,
        "compatibility_digest": "d",
    }


def test_v2_claim_spec_carries_bytes_and_state_agent(v2_claim, service_socket):
    assert v2_claim["service_socket_path"] == service_socket
    assert v2_claim["gaas"] == {
        "tenant": "t1",
        "lineage": "dgd-a",
        "total_bytes": _GIB,
        "minimum_bytes": _GIB // 2,
        "maximum_bytes": 2 * _GIB,
        "state_agent": {"namespace": "ns", "component": "backend"},
    }


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tenant": "t"}, "set together"),
        ({"lineage": "l"}, "set together"),
        ({"minimum_gb": 1}, "need tenant and lineage"),
        ({"service_socket": "/s/x.sock"}, "need tenant and lineage"),
        ({"tenant": "t", "lineage": "l"}, "service socket"),
        (
            {"tenant": "t", "lineage": "l", "service_socket": "rel.sock"},
            "absolute",
        ),
        (
            {"tenant": "t", "lineage": "l", "service_socket": "/s/x.sock",
             "maximum_gb": 0.5},
            "minimum <= size <= maximum",
        ),
        (
            {"tenant": "t", "lineage": "l", "service_socket": "/s/x.sock",
             "minimum_gb": 2},
            "minimum <= size <= maximum",
        ),
        (
            {"tenant": "t", "lineage": "l", "service_socket": "/s/x.sock",
             "maximum_gb": 257},
            "maximum_gb must be 0.5 to 256 GiB in whole MiB",
        ),
    ],
)
def test_v2_claim_spec_rejects_bad_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        claim_spec("pool", 1.0, "d", **kwargs)


def _v2_args(command, claim_file, service_socket, *extra):
    return [
        command,
        "--claim-file",
        str(claim_file),
        "--pool-id",
        "pool-a",
        "--size-gb",
        "1.0",
        "--compatibility-digest",
        "model-layout-digest",
        *_V2_FLAGS,
        "--service-socket",
        service_socket,
        *extra,
    ]


def test_v2_init_waits_for_the_socket_not_the_broker(
    tmp_path, service_socket, v2_claim
):
    output = tmp_path / "claim.json"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(service_socket)
        assert (
            main(
                _v2_args("init", output, service_socket, "--timeout-s", "5"),
                ensure_pool=lambda *a, **k: pytest.fail("v2 must not call the broker"),
            )
            == 0
        )
    descriptor = json.loads(output.read_text())
    assert descriptor == {
        **v2_claim,
        "state": "ready",
        "control_port": 55001,
        "guard_index": 0,
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_wait_for_socket_times_out_without_one(tmp_path):
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(TimeoutError, match="did not appear"):
        wait_for_socket(
            str(tmp_path / "none.sock"), 3, sleep=sleep, clock=lambda: now[0]
        )
    assert now[0] == 3
    (tmp_path / "file").write_text("")
    with pytest.raises(TimeoutError):  # a regular file is not the supervisor
        wait_for_socket(
            str(tmp_path / "file"), 1, sleep=sleep, clock=lambda: now[0]
        )


def test_v1_still_requires_the_broker_socket(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "init",
                "--claim-file",
                str(tmp_path / "c.json"),
                "--pool-id",
                "pool-a",
                "--size-gb",
                "1",
                "--compatibility-digest",
                "d",
            ],
            ensure_pool=lambda *a, **k: pytest.fail("unexpected"),
        )


def test_v2_launch_passes_gaas_settings_without_a_holder_address(
    tmp_path, service_socket, v2_claim, monkeypatch
):
    descriptor = {**v2_claim, "state": "ready", "control_port": 55001, "guard_index": 0}
    claim_file = tmp_path / "claim.json"
    claim_file.write_text(json.dumps(descriptor))
    calls = []
    monkeypatch.setattr(os, "execv", lambda exe, args: calls.append(args))
    assert (
        main(
            _v2_args(
                "launch",
                claim_file,
                service_socket,
                "--cpu-bytes",
                "4096",
                "--",
                "--model",
                "m",
            ),
            ensure_pool=lambda *a, **k: pytest.fail("unexpected broker call"),
        )
        == 0
    )
    config = json.loads(calls[0][-1])
    tier = config["kv_connector_extra_config"]["secondary_tiers"][0]
    assert tier["gaas"] == v2_claim["gaas"]
    assert tier["kvcr_service_socket_path"] == service_socket
    assert tier["control_ports"] == [55001]
    assert "control_advertise_host" not in tier


def test_v2_launch_rejects_a_descriptor_for_other_settings(v2_claim):
    descriptor = {**v2_claim, "state": "ready", "control_port": 55001}
    validate_gaas_descriptor(descriptor, v2_claim)
    stale = {**descriptor, "gaas": {**v2_claim["gaas"], "lineage": "other"}}
    with pytest.raises(ValueError, match="gaas does not match"):
        validate_gaas_descriptor(stale, v2_claim)
    with pytest.raises(ValueError, match="not ready"):
        validate_gaas_descriptor({**descriptor, "state": "pending"}, v2_claim)


def test_v2_launch_rejects_conflicting_tier_gaas(v2_claim):
    descriptor = {**v2_claim, "state": "ready", "control_port": 55001}
    conflicting = json.dumps(
        {
            "kv_connector_extra_config": {
                "secondary_tiers": [{"type": "kvcr", "gaas": {"tenant": "x"}}]
            }
        }
    )
    with pytest.raises(ValueError, match="gaas conflicts"):
        launch_argv(descriptor, 4096, ["--kv-transfer-config", conflicting])


# -- probe: the claimant through the KVCR client, with a fake client.

if sys.platform == "linux":

    class _FakeHold:
        def __init__(self, calls):
            self.calls = calls
            self.slot_name, self.claim_generation = "t/l/0", 3
            self.layout_generation = 1
            self.region_fds = [os.memfd_create("probe")]
            os.ftruncate(self.region_fds[0], 2 * mmap.PAGESIZE)
            self.regions = (PoolRegion(0, 0, mmap.PAGESIZE, None),)

        def release(self, *, activated=True):
            self.calls.append(("release", activated))

        def close_local(self):
            self.calls.append(("close_local",))
            for fd in self.region_fds:
                os.close(fd)
            self.region_fds = []

    class _FakeClient:
        calls: list = []

        def __init__(self, socket_path):
            self.calls.append(("client", socket_path))

        def claim(self, *args, **kwargs):
            self.calls.append(("claim", args, kwargs["gaas"].tenant))
            return _FakeHold(self.calls)

        def grow(self, slot_name, additional_bytes, claim_generation):
            self.calls.append(("grow", slot_name, additional_bytes, claim_generation))
            return _Grown(1, (PoolRegion(1, mmap.PAGESIZE, mmap.PAGESIZE, None),), 2)

        def shrink(self, slot_name, region_indices, claim_generation):
            self.calls.append(("shrink", slot_name, tuple(region_indices)))
            return _Shrunk(1, tuple(region_indices), 3)


_PROBE_FLAGS = [
    "--socket", "/x/service.sock", "--tenant", "t", "--lineage", "l",
    "--compatibility-digest", "d", "--tier-config", "pool0:1024",
    "--total-bytes", "4096", "--minimum-bytes", "4096", "--maximum-bytes", "8192",
    "--grow-bytes", "4096", "--mode", "grow-shrink", "--release",
]  # fmt: skip


@pytest.mark.skipif(sys.platform != "linux", reason="memfd")
def test_probe_emits_the_claimants_lines_through_the_client_methods(
    monkeypatch, capsys
):
    from kvcr.service import claimant

    monkeypatch.setattr(claimant, "KVCRClient", _FakeClient)
    _FakeClient.calls = []
    claimant_code = claimant.main(_PROBE_FLAGS)
    claimant_out = capsys.readouterr().out
    claimant_calls, _FakeClient.calls = _FakeClient.calls, []
    probe_code = main(["probe", *_PROBE_FLAGS], ensure_pool=lambda *a, **k: None)
    probe_out = capsys.readouterr().out
    assert claimant_code == probe_code == 0
    assert probe_out == claimant_out
    steps = [json.loads(line) for line in probe_out.splitlines()]
    assert [step["step"] for step in steps] == [
        "granted", "written", "grown", "shrunk", "released"
    ]  # fmt: skip
    assert steps[0]["layout_generation"] == 1 and steps[0]["bytes"] == mmap.PAGESIZE
    assert steps[2]["layout_generation"] == 2
    assert steps[3] == {"step": "shrunk", "region_indices": [1], "layout_generation": 3}
    assert _FakeClient.calls == claimant_calls
    kinds = [call[0] for call in _FakeClient.calls]
    assert kinds == ["client", "claim", "grow", "shrink", "release", "close_local"]
    assert _FakeClient.calls[2] == ("grow", "t/l/0", 4096, 3)


@pytest.mark.skipif(sys.platform != "linux", reason="memfd")
def test_probe_error_line_and_exit_code_match_the_claimant(monkeypatch, capsys):
    from kvcr.service import claimant

    class Refusing(_FakeClient):
        def claim(self, *args, **kwargs):
            raise KVCRServiceReasonError("no room", "capacity_below_minimum")

    monkeypatch.setattr(claimant, "KVCRClient", Refusing)
    claimant_code = claimant.main(_PROBE_FLAGS)
    claimant_out = capsys.readouterr().out
    probe_code = main(["probe", *_PROBE_FLAGS], ensure_pool=lambda *a, **k: None)
    probe_out = capsys.readouterr().out
    assert claimant_code == probe_code == 2
    assert probe_out == claimant_out
    assert json.loads(probe_out) == {
        "step": "error",
        "reason": "capacity_below_minimum",
        "message": "no room",
    }
