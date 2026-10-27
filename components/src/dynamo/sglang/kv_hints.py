# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward router KV hints and advertise KVCR HiCache sources to Dynamo."""

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
from dynamo.sglang.storage_config import parse_hicache_storage_extra_config


def kv_hint_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Pass the private router hint only when the engine accepts ``kv_hints``."""
    hint = request.get("kv_hint")
    if not isinstance(hint, Mapping):
        return {}
    return filter_supported_async_generate_kwargs(engine, {"kv_hints": dict(hint)})


def kvcr_hint_endpoints(
    *,
    extra_config: Mapping[str, Any],
    dp_size: int,
    tp_size: int,
    attn_cp_size: int,
    pp_size: int,
) -> dict[str, str]:
    """One base endpoint per attention-DP rank in SGLang's rank-port layout."""
    host = extra_config.get("control_advertise_host")
    port = extra_config.get("control_port", 32100)
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
    if type(port) is not int or min(dp_size, tp_size, attn_cp_size, pp_size) < 1:
        raise ValueError("KVCR control-port range is invalid")
    stride = tp_size * attn_cp_size * pp_size
    last_port = port + dp_size * stride - 1
    if not (0 < port <= 65535 and 0 < last_port <= 65535):
        raise ValueError("KVCR control-port range is invalid")
    return {
        str(rank): f"tcp://{host}:{port + rank * stride}" for rank in range(dp_size)
    }


def publish_kvcr_hint_capabilities(
    runtime_config, engine, server_args, worker_type
) -> None:
    """Advertise supported single-host KVCR HiCache workers with explicit hosts."""
    server_args = resolved_server_args(server_args)
    if (
        not getattr(server_args, "enable_hierarchical_cache", False)
        or getattr(server_args, "hicache_storage_backend", None) != "kvcr"
        or getattr(server_args, "nnodes", 1) != 1
        or getattr(server_args, "pp_size", 1) != 1
        or not filter_supported_async_generate_kwargs(engine, {"kv_hints": None})
    ):
        return
    extra = parse_hicache_storage_extra_config(
        server_args.hicache_storage_backend_extra_config, strict=True
    )
    if (
        extra.get("enable_remote_hint", True) is not True
        or extra.get("control_advertise_host") is None
    ):
        # Local-only workers still register, but must not advertise an implicit
        # loopback endpoint for peers on other hosts.
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
    )
    for key, value in (
        (KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, str(worker_type)),
        (KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, endpoints),
        (KV_HINT_TRANSFER_CAPABILITY_KEY, True),
    ):
        runtime_config.set_engine_specific(key, json.dumps(value))
