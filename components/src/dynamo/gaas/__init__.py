# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Init-container gate and launcher for a single-worker KVCR pool claim."""

import argparse
import ipaddress
import json
import os
import re
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

Claim = dict[str, Any]
EnsurePool = Callable[..., Claim]


def claim_spec(pool_id: str, size_gb: float, compatibility_digest: str) -> Claim:
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", pool_id) is None:
        raise ValueError("pool_id must be a DNS label")
    try:
        size = Decimal(str(size_gb))
    except InvalidOperation as error:
        raise ValueError("invalid size_gb") from error
    if (
        not size.is_finite()
        or not Decimal("0.5") <= size <= 256
        or size != Decimal(int(size * 1024)) / 1024
    ):
        raise ValueError("size_gb must be 0.5 to 256 GiB in whole MiB")
    if not isinstance(compatibility_digest, str) or not compatibility_digest.strip():
        raise ValueError("compatibility_digest must be non-empty")
    return {
        "pool_id": pool_id,
        "size_gb": float(size_gb),
        "compatibility_digest": compatibility_digest,
    }


def validate_descriptor(
    descriptor: Claim, expected: Claim, broker_socket: str
) -> Claim:
    if not isinstance(descriptor, dict):
        raise ValueError("claim descriptor must be an object")
    for key, value in expected.items():
        if descriptor.get(key) != value:
            raise ValueError(f"claim descriptor {key} does not match the request")
    if isinstance(descriptor["size_gb"], bool):
        raise ValueError("claim descriptor size_gb must be numeric")
    if descriptor.get("state") != "ready":
        raise ValueError("claim descriptor is not ready")
    holder_uid = descriptor.get("holder_uid")
    if not isinstance(holder_uid, str) or str(uuid.UUID(holder_uid)) != holder_uid:
        raise ValueError("claim descriptor requires a canonical holder UID")
    holder_ip = descriptor.get("holder_pod_ip")
    if not isinstance(holder_ip, str):
        raise ValueError("claim descriptor requires a holder Pod IP")
    address = ipaddress.ip_address(holder_ip)
    if address.is_unspecified or address.is_loopback or address.is_multicast:
        raise ValueError("holder Pod IP must be a routable unicast address")
    if descriptor.get("control_port") != 55001 or isinstance(
        descriptor.get("control_port"), bool
    ):
        raise ValueError("claim descriptor requires holder control port 55001")
    if descriptor.get("guard_index") != 0 or isinstance(
        descriptor.get("guard_index"), bool
    ):
        raise ValueError("claim descriptor requires guard index zero")
    socket_path = descriptor.get("service_socket_path")
    if not isinstance(socket_path, str):
        raise ValueError("claim descriptor requires a service socket path")
    endpoint = Path(socket_path)
    root = Path(broker_socket).parent
    if (
        not Path(broker_socket).is_absolute()
        or not endpoint.is_absolute()
        or ".." in endpoint.parts
        or len(os.fsencode(socket_path)) >= 108
        or not endpoint.resolve().is_relative_to(root.resolve())
        or endpoint.resolve() == Path(broker_socket).resolve()
    ):
        raise ValueError("service socket must be inside the broker socket directory")
    return descriptor


def prepare_claim(
    broker_socket: str,
    expected: Claim,
    claim_file: Path,
    timeout_s: float,
    ensure_pool: EnsurePool,
) -> Claim:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    descriptor = validate_descriptor(
        ensure_pool(broker_socket, expected, timeout_s=timeout_s),
        expected,
        broker_socket,
    )
    claim_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=claim_file.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
            os.chmod(temporary_path, 0o600)
            json.dump(descriptor, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, claim_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return descriptor


def _canonical_backend_args(argv: Sequence[str]) -> list[str]:
    guarded = (
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
        "--kv-transfer-config",
    )
    result = []
    for argument in argv:
        if argument == "--":
            raise ValueError("GaaS backend arguments cannot contain another --")
        option, separator, value = argument.partition("=")
        root, dot, _ = option.partition(".")
        if root.startswith("--"):
            canonical = root.replace("_", "-")
            if canonical == "--config":
                raise ValueError(
                    "GaaS requires explicit backend arguments, not --config"
                )
            if any(name.startswith(canonical) for name in guarded):
                # vLLM appends dotted JSON after ordinary CLI values. Do not let
                # it replace the validated configuration appended below.
                if dot or canonical not in guarded:
                    raise ValueError(
                        "GaaS requires complete guarded option names and whole JSON"
                    )
                argument = canonical + separator + value
        elif root in ("-t", "-p", "-d") or (
            dot and any(root.startswith(alias) for alias in ("-tp", "-pp", "-dp"))
        ):
            raise ValueError("GaaS requires complete guarded short option names")
        result.append(argument)
    return result


def _extract_transfer_config(argv: Sequence[str]) -> tuple[list[str], Claim]:
    remaining: list[str] = []
    config: Claim = {}
    seen = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--kv-transfer-config" or argument.startswith(
            "--kv-transfer-config="
        ):
            if seen:
                raise ValueError("only one --kv-transfer-config is supported")
            seen = True
            if argument == "--kv-transfer-config":
                index += 1
                if index == len(argv):
                    raise ValueError("--kv-transfer-config requires JSON")
                raw = argv[index]
            else:
                raw = argument.split("=", 1)[1]
            config = json.loads(raw)
            if not isinstance(config, dict):
                raise ValueError("--kv-transfer-config must be a JSON object")
        else:
            remaining.append(argument)
        index += 1
    return remaining, config


def launch_argv(
    descriptor: Claim, cpu_bytes: int, backend_args: Sequence[str]
) -> list[str]:
    if isinstance(cpu_bytes, bool) or cpu_bytes <= 0:
        raise ValueError("cpu_bytes must be positive")
    backend_args = _canonical_backend_args(backend_args)
    topology = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name, short in (
        ("tensor-parallel-size", "-tp"),
        ("pipeline-parallel-size", "-pp"),
        ("data-parallel-size", "-dp"),
    ):
        topology.add_argument(f"--{name}", short, type=int, default=1)
    parallel, _ = topology.parse_known_args(backend_args)
    if any(value != 1 for value in vars(parallel).values()):
        raise ValueError("GaaS launch supports TP=PP=DP=1 only")
    remaining, config = _extract_transfer_config(backend_args)
    if config.get("kv_connector", "OffloadingConnector") != "OffloadingConnector":
        raise ValueError("GaaS requires OffloadingConnector")
    if config.get("kv_connector_module_path") is not None:
        raise ValueError("GaaS requires the built-in OffloadingConnector")
    if config.get("kv_role", "kv_both") != "kv_both":
        raise ValueError("GaaS requires kv_role=kv_both")
    extra = config.setdefault("kv_connector_extra_config", {})
    if not isinstance(extra, dict):
        raise ValueError("kv_connector_extra_config must be an object")
    if extra.get("spec_name", "TieringOffloadingSpec") != "TieringOffloadingSpec":
        raise ValueError("GaaS requires TieringOffloadingSpec")
    if extra.get("cpu_bytes_to_use", cpu_bytes) != cpu_bytes:
        raise ValueError("cpu_bytes conflicts with existing offload configuration")
    tiers = extra.setdefault("secondary_tiers", [{"type": "kvcr"}])
    if (
        not isinstance(tiers, list)
        or len(tiers) != 1
        or not isinstance(tiers[0], dict)
        or tiers[0].get("type") != "kvcr"
    ):
        raise ValueError("GaaS requires exactly one KVCR secondary tier")
    tier = tiers[0]
    required = {
        "kvcr_service_socket_path": descriptor["service_socket_path"],
        "compatibility_digest": descriptor["compatibility_digest"],
        "control_host": "0.0.0.0",
        "control_ports": [descriptor["control_port"]],
        "control_advertise_host": descriptor["holder_pod_ip"],
    }
    for key, value in required.items():
        if key in tier and tier[key] != value:
            raise ValueError(f"existing secondary-tier {key} conflicts with the claim")
        tier[key] = value
    capabilities = tier.setdefault("router_capabilities", ["router_hint"])
    if not isinstance(capabilities, list) or "router_hint" not in capabilities:
        raise ValueError("KVCR requires router_hint capability")
    extra["spec_name"] = "TieringOffloadingSpec"
    extra["cpu_bytes_to_use"] = cpu_bytes
    extra["self_describing_kv_events"] = True
    config["kv_connector"] = "OffloadingConnector"
    config["kv_role"] = "kv_both"
    return [
        sys.executable,
        "-m",
        "dynamo.vllm",
        *remaining,
        "--kv-transfer-config",
        json.dumps(config, separators=(",", ":")),
    ]


def main(argv: Sequence[str] | None = None, *, ensure_pool: EnsurePool) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "launch"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--broker-socket", required=True)
        command_parser.add_argument("--claim-file", type=Path, required=True)
        command_parser.add_argument("--pool-id", required=True)
        command_parser.add_argument("--size-gb", type=float, required=True)
        command_parser.add_argument("--compatibility-digest", required=True)
        if command == "init":
            command_parser.add_argument("--timeout-s", type=float, default=180)
        else:
            command_parser.add_argument("--cpu-bytes", type=int, required=True)
            command_parser.add_argument("backend_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    expected = claim_spec(args.pool_id, args.size_gb, args.compatibility_digest)
    if args.command == "init":
        prepare_claim(
            args.broker_socket,
            expected,
            args.claim_file,
            args.timeout_s,
            ensure_pool,
        )
        return 0
    with args.claim_file.open(encoding="utf-8") as input_file:
        descriptor = validate_descriptor(
            json.load(input_file), expected, args.broker_socket
        )
    backend_args = args.backend_args
    if backend_args[:1] == ["--"]:
        backend_args = backend_args[1:]
    command = launch_argv(descriptor, args.cpu_bytes, backend_args)
    os.execv(command[0], command)
    return 0
