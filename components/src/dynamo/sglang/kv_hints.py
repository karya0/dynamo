# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVCR transfer capability and in-process request-hint forwarding."""

from __future__ import annotations

import ipaddress
import json
import socket
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


def kv_hint_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Dynamo's private kv_hint envelope maps to SGLang's kv_hints input.

    The router creates the envelope after worker selection; do not pull a
    client-supplied hint out of extra_args or nvext. Older engines can ignore
    this advisory input, but must not advertise transfer capability.
    """
    hint = request.get("kv_hint")
    if not isinstance(hint, Mapping):
        return {}
    return filter_supported_async_generate_kwargs(engine, {"kv_hints": dict(hint)})


# SGLang's own control-plane defaults for each KVCR mode.
_KVCR_DEFAULT_CONTROL_PORT = {"hicache": 32100, "linker": 19500}


def kvcr_mode(server_args: Any) -> str | None:
    """The direct linker or the HiCache storage backend; None without KVCR.

    SGLang rejects enabling both, and both read the KVCR options from
    --hicache-storage-backend-extra-config.
    """
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
    # The linker names the option advertise_host and, when unset, advertises
    # the address of this host's name; the in-process engine shares the host.
    host = extra_config.get("control_advertise_host")
    if host is None:
        host = extra_config.get("advertise_host")
    if host is None:
        host = (
            socket.gethostbyname(socket.gethostname())
            if mode == "linker"
            else "127.0.0.1"
        )
    port = extra_config.get("control_port", _KVCR_DEFAULT_CONTROL_PORT[mode])
    if not isinstance(host, str) or not host:
        raise ValueError("KVCR requires an advertisable control host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
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
        if extra.startswith("@"):
            # The SGLang controller resolves file configs. Avoid advertising a
            # guessed endpoint until registration can read its resolved result.
            raise ValueError("KVCR hint publication requires inline JSON configuration")
        extra = json.loads(extra)
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise ValueError("KVCR HiCache extra configuration must be an object")
    # The linker has no such option and rejects unknown keys; it is enabled.
    if mode == "hicache" and extra.get("enable_remote_hint", True) is not True:
        return
    # WorkerType is a PyO3 enum, not enum.Enum: its wire role is __str__,
    # and accessing .value aborts registration after the model has loaded.
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
