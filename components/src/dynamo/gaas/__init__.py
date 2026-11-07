# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Init-container gate and launcher for a single-worker KVCR pool claim."""

import argparse
import ipaddress
import json
import os
import re
import sys
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

Claim = dict[str, Any]
EnsurePool = Callable[..., Claim]


def _capacity(value: float | str, name: str) -> Decimal:
    try:
        size = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"invalid {name}") from error
    if (
        not size.is_finite()
        or not Decimal("0.5") <= size <= 256
        or size != Decimal(int(size * 1024)) / 1024
    ):
        raise ValueError(f"{name} must be 0.5 to 256 GiB in whole MiB")
    return size


def _socket_path(path: str) -> str:
    endpoint = Path(path)
    if (
        not endpoint.is_absolute()
        or ".." in endpoint.parts
        or len(os.fsencode(path)) >= 108
    ):
        raise ValueError("service socket must be an absolute path under 108 bytes")
    return path


def claim_spec(
    pool_id: str,
    size_gb: float,
    compatibility_digest: str,
    *,
    tenant: str | None = None,
    lineage: str | None = None,
    minimum_gb: float | None = None,
    maximum_gb: float | None = None,
    state_agent_namespace: str | None = None,
    state_agent_component: str | None = None,
    service_socket: str | None = None,
) -> Claim:
    """The claim to make. With tenant and lineage it is a GaaS (v2) claim."""
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", pool_id) is None:
        raise ValueError("pool_id must be a DNS label")
    size = _capacity(size_gb, "size_gb")
    if not isinstance(compatibility_digest, str) or not compatibility_digest.strip():
        raise ValueError("compatibility_digest must be non-empty")
    spec: Claim = {
        "pool_id": pool_id,
        "size_gb": float(size_gb),
        "compatibility_digest": compatibility_digest,
    }
    optional = (
        minimum_gb,
        maximum_gb,
        state_agent_namespace,
        state_agent_component,
    )
    if tenant is None and lineage is None:
        if service_socket is not None or any(v is not None for v in optional):
            raise ValueError("GaaS settings need tenant and lineage")
        return spec
    for name, value in (("tenant", tenant), ("lineage", lineage)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"GaaS {name} must be set together and non-empty")
    if service_socket is None:
        raise ValueError("a GaaS claim needs the service socket")
    gaas: Claim = {
        "tenant": tenant,
        "lineage": lineage,
        "total_bytes": int(size * (1 << 30)),
    }
    for name, value in (("minimum", minimum_gb), ("maximum", maximum_gb)):
        if value is not None:
            gaas[f"{name}_bytes"] = int(_capacity(value, f"{name}_gb") * (1 << 30))
    if gaas.get("minimum_bytes", 0) > gaas["total_bytes"] or gaas[
        "total_bytes"
    ] > gaas.get("maximum_bytes", gaas["total_bytes"]):
        raise ValueError("GaaS capacities must satisfy minimum <= size <= maximum")
    state_agent = {
        key: value
        for key, value in (
            ("namespace", state_agent_namespace),
            ("component", state_agent_component),
        )
        if value is not None
    }
    if state_agent:
        gaas["state_agent"] = state_agent
    spec["service_socket_path"] = _socket_path(service_socket)
    spec["gaas"] = gaas
    return spec


def wait_for_socket(
    path: str,
    timeout_s: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Block until a socket exists at ``path``: the supervisor is serving claims."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = clock() + timeout_s
    while True:
        try:
            if stat.S_ISSOCK(os.stat(path).st_mode):
                return
        except FileNotFoundError:
            pass
        if clock() >= deadline:
            raise TimeoutError(f"service socket {path} did not appear")
        sleep(1)


def validate_gaas_descriptor(descriptor: Claim, expected: Claim) -> Claim:
    if not isinstance(descriptor, dict):
        raise ValueError("claim descriptor must be an object")
    for key, value in expected.items():
        if descriptor.get(key) != value:
            raise ValueError(f"claim descriptor {key} does not match the request")
    if descriptor.get("state") != "ready":
        raise ValueError("claim descriptor is not ready")
    port, index = descriptor.get("control_port"), descriptor.get("guard_index")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError("claim descriptor requires a valid control port")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("claim descriptor requires a valid guard index")
    return descriptor


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
    _write_claim_file(descriptor, claim_file)
    return descriptor


def _write_claim_file(descriptor: Claim, claim_file: Path) -> None:
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
    descriptor: Claim,
    cpu_bytes: int,
    backend_args: Sequence[str],
    *,
    advertise_host: str | None = None,
    state_agent: bool = False,
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
    }
    if "gaas" in descriptor:  # v2: the holder is created by the claim itself
        if not advertise_host:
            raise SystemExit(
                "GaaS v2 needs --advertise-host/GAAS_ADVERTISE_HOST: "
                "the router hint source must be advertisable"
            )
        required["gaas"] = descriptor["gaas"]
        required["control_host"] = advertise_host
        required["control_advertise_host"] = advertise_host
        required["guard_index_base"] = descriptor["guard_index"]
    else:
        required["control_advertise_host"] = descriptor["holder_pod_ip"]
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
    if state_agent:
        slot = descriptor["guard_index"]
        extra["dynamo_state_agent"] = {
            "raw_advertise_host": advertise_host,
            "cache_owner_ids": {
                "0": "01010101010101010101010101010101:"
                f"02020202020202020202020202020202/0303030303030303/{slot:032x}"
            },
        }
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


def _guard_index(value: str) -> int:
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return index


def main(argv: Sequence[str] | None = None, *, ensure_pool: EnsurePool) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["probe"]:
        # Same flags, JSON lines and exit codes as python -m kvcr.service.claimant,
        # through the KVCRClient methods the engine uses.
        from kvcr.service.claimant import main as probe

        return probe(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "probe", help="CPU-only claimant; flags as kvcr.service.claimant"
    )
    for command in ("init", "launch"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("--broker-socket")
        command_parser.add_argument("--claim-file", type=Path, required=True)
        command_parser.add_argument("--pool-id", required=True)
        command_parser.add_argument("--size-gb", required=True)
        command_parser.add_argument("--compatibility-digest", required=True)
        # GaaS (v2) claim: selected by --tenant and --lineage, size-gb is the total.
        command_parser.add_argument("--tenant")
        command_parser.add_argument("--lineage")
        command_parser.add_argument("--minimum-gb")
        command_parser.add_argument("--maximum-gb")
        command_parser.add_argument("--state-agent-namespace")
        command_parser.add_argument("--state-agent-component")
        command_parser.add_argument("--service-socket")
        command_parser.add_argument("--guard-index", type=_guard_index, default=0)
        command_parser.add_argument("--control-port", type=int)
        command_parser.add_argument(
            "--advertise-host", default=os.environ.get("GAAS_ADVERTISE_HOST")
        )
        if command == "init":
            command_parser.add_argument("--timeout-s", type=float, default=180)
        else:
            command_parser.add_argument("--cpu-bytes", type=int, required=True)
            command_parser.add_argument("--state-agent", action="store_true")
            command_parser.add_argument("backend_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    expected = claim_spec(
        args.pool_id,
        args.size_gb,
        args.compatibility_digest,
        tenant=args.tenant,
        lineage=args.lineage,
        minimum_gb=args.minimum_gb,
        maximum_gb=args.maximum_gb,
        state_agent_namespace=args.state_agent_namespace,
        state_agent_component=args.state_agent_component,
        service_socket=args.service_socket,
    )
    gaas = "gaas" in expected
    control_port = args.control_port or 55001 + args.guard_index
    if not gaas and args.broker_socket is None:
        parser.error("--broker-socket is required without --tenant and --lineage")
    if args.command == "init":
        if gaas:
            # The supervisor creates the holder inside the claim: wait for its socket.
            wait_for_socket(expected["service_socket_path"], args.timeout_s)
            _write_claim_file(
                {
                    **expected,
                    "state": "ready",
                    "control_port": control_port,
                    "guard_index": args.guard_index,
                },
                args.claim_file,
            )
            return 0
        prepare_claim(
            args.broker_socket,
            expected,
            args.claim_file,
            args.timeout_s,
            ensure_pool,
        )
        return 0
    with args.claim_file.open(encoding="utf-8") as input_file:
        loaded = json.load(input_file)
    descriptor = (
        validate_gaas_descriptor(loaded, expected)
        if gaas
        else validate_descriptor(loaded, expected, args.broker_socket)
    )
    backend_args = args.backend_args
    if backend_args[:1] == ["--"]:
        backend_args = backend_args[1:]
    command = launch_argv(
        descriptor,
        args.cpu_bytes,
        backend_args,
        advertise_host=args.advertise_host,
        state_agent=args.state_agent,
    )
    os.execv(command[0], command)
    return 0
