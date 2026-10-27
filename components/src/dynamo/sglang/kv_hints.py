# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward router KV hints and advertise KVCR sources to Dynamo."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from typing import Any

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.sglang._compat import (
    filter_supported_async_generate_kwargs,
    resolved_server_args,
)
from dynamo.sglang.storage_config import load_hicache_storage_extra_config_file


def kv_hint_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Pass the private router hint only when the engine accepts ``kv_hints``."""
    hint = request.get("kv_hint")
    if not isinstance(hint, Mapping):
        return {}
    return filter_supported_async_generate_kwargs(engine, {"kv_hints": dict(hint)})


def kvcr_mode(server_args: Any) -> str | None:
    """Return the active KVCR integration mode, if any."""
    if (
        getattr(server_args, "enable_unified_cache_external_linker", False)
        and getattr(server_args, "unified_cache_external_linker_backend", None)
        == "kvcr"
    ):
        return "linker"
    if (
        getattr(server_args, "enable_hierarchical_cache", False)
        and getattr(server_args, "hicache_storage_backend", None) == "kvcr"
    ):
        return "hicache"
    return None


def kvcr_hint_endpoints(
    *,
    extra_config: Mapping[str, Any],
    dp_size: int,
    tp_size: int,
    attn_cp_size: int,
    pp_size: int,
    mode: str = "hicache",
) -> dict[str, str]:
    """One base endpoint per DP replica, matching SGLang's rank-port layout."""
    if mode not in ("hicache", "linker"):
        raise ValueError(f"Unknown KVCR mode: {mode}")
    # HiCache does not consume advertise_host: accepting it here would publish
    # an address different from the storage backend's actual bound endpoint.
    host = extra_config.get("control_advertise_host", "127.0.0.1")
    port = extra_config.get("control_port", 32100)
    if mode == "linker":
        # Linker defaults differ from HiCache; require explicit coordinates.
        host = extra_config.get("advertise_host")
        alternate = extra_config.get("control_advertise_host")
        if host is not None and alternate is not None and host != alternate:
            raise ValueError("Conflicting KVCR linker advertise hosts")
        if host is None:
            host = alternate
        port = extra_config.get("control_port")
    if not isinstance(host, str) or not host:
        raise ValueError("KVCR requires an advertisable control host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
        if any(char.isspace() or char in "/@?#:[]" for char in host):
            raise ValueError("KVCR requires a host, not a URL or malformed address")
    if address is not None:
        if address.is_unspecified:
            raise ValueError("KVCR cannot advertise an unspecified control host")
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    stride = tp_size * attn_cp_size * pp_size
    if (
        type(port) is not int
        or min(dp_size, tp_size, attn_cp_size, pp_size) < 1
        or not 0 < port <= port + dp_size * stride - 1 <= 65535
    ):
        raise ValueError("KVCR control-port range is invalid")
    return {
        str(rank): f"tcp://{host}:{port + rank * stride}" for rank in range(dp_size)
    }


def publish_kvcr_hint_capabilities(
    runtime_config, engine, server_args, worker_type
) -> None:
    """Opt in only for the supported single-host KVCR linker or HiCache path."""
    server_args = resolved_server_args(server_args)
    mode = kvcr_mode(server_args)
    if (
        engine is None
        or mode is None
        or getattr(server_args, "nnodes", 1) != 1
        or getattr(server_args, "pp_size", 1) != 1
        or not filter_supported_async_generate_kwargs(engine, {"kv_hints": None})
    ):
        return
    extra = server_args.hicache_storage_backend_extra_config
    if isinstance(extra, str):
        extra = extra.strip()
        if not extra:
            extra = {}
        elif extra.startswith("@"):
            extra = load_hicache_storage_extra_config_file(extra[1:])
        else:
            extra = json.loads(extra)
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise ValueError("KVCR HiCache extra configuration must be an object")
    # Legacy linker revisions have no switch; newer ones can disable hints.
    if extra.get("enable_remote_hint", True) is not True:
        return
    if (
        mode == "linker"
        and extra.get("advertise_host") is None
        and extra.get("enable_remote_hint", False) is not True
    ):
        # The newer control_advertise_host dialect defaults remote hints off.
        # The legacy advertise_host runtime has no switch and enables peers.
        return
    if mode == "linker" and (
        "control_port" not in extra
        or (
            extra.get("advertise_host") is None
            and extra.get("control_advertise_host") is None
        )
    ):
        # Local-only/default linkers can still serve, but cannot advertise a
        # guessed source-control endpoint to another worker.
        return
    # WorkerType is a PyO3 enum; str() gives its wire role.
    role = str(worker_type)
    if role == "agg":
        role = "aggregated"
    if role not in ("aggregated", "prefill", "decode"):
        return
    # DP attention partitions TP into independent attention/cache groups.
    dp_size = server_args.attn_dp_size
    attn_cp_size = server_args.attn_cp_size
    tp_size = server_args.tp_size // (dp_size * attn_cp_size)
    endpoints = kvcr_hint_endpoints(
        extra_config=extra,
        dp_size=dp_size,
        tp_size=tp_size,
        attn_cp_size=attn_cp_size,
        pp_size=server_args.pp_size,
        mode=mode,
    )
    for key, value in (
        (KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, role),
        (KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, endpoints),
        (KV_HINT_TRANSFER_CAPABILITY_KEY, True),
    ):
        runtime_config.set_engine_specific(key, json.dumps(value))
