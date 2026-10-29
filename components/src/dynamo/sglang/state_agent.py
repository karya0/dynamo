# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach actual, process-local HiCache backend inventory to the KV state agent."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse, urlsplit

from dynamo.common.constants import DisaggregationMode
from dynamo.common.kv_state_agent import (
    CACHE_OWNER_PATTERN,
    bounded_close,
    validate_advertise_host,
)
from dynamo.llm import KvStateAttachmentOwner
from dynamo.sglang._compat import resolved_server_args
from dynamo.sglang.capacity import kv_event_block_size, model_card_dp_rank_bounds
from dynamo.sglang.gateway import effective_gateway_workers
from dynamo.sglang.kv_hints import (
    kvcr_hint_endpoints,
    kvcr_linker_enabled,
    reported_kvcr_hint_endpoints,
)

logger = logging.getLogger(__name__)


def inventory_sources(engine) -> dict[int, Mapping[str, object]]:
    """Read capabilities from scheduler metadata, never backend configuration."""
    initialization = getattr(engine, "_scheduler_init_result", None)
    if initialization is None:
        return {}
    sources: dict[int, Mapping[str, object]] = {}
    for info in initialization.scheduler_infos:
        hints = info.get("kv_hint_sources", [])
        if not isinstance(hints, list):
            raise TypeError("SGLang KV hint sources must be a list")
        for source in hints:
            if not isinstance(source, Mapping):
                raise TypeError("SGLang KV hint source must be an object")
            capabilities = source.get("capabilities", [])
            if not isinstance(capabilities, list):
                raise TypeError("SGLang KV hint capabilities must be a list")
            if "kvcr_inventory" not in capabilities:
                continue
            rank = source.get("dp_rank")
            if type(rank) is not int or rank < 0:
                raise ValueError("KVCR inventory source requires a nonnegative DP rank")
            if rank in sources and sources[rank] != source:
                raise ValueError("Conflicting KVCR inventory sources for one DP rank")
            sources[rank] = source
    return sources


def validate_state_agent_worker(config: Any) -> str | None:
    """Validate the direct-linker opt-in (--kvcr-state-agent-owner) at launch.

    Returns the owner when the flag is set, None otherwise. This does not force
    SGLang's engine resolution; HiCache inventory is validated separately in
    state_agent_descriptors once the scheduler has reported its sources.
    """
    dynamo_args = config.dynamo_args
    owner = getattr(dynamo_args, "kvcr_state_agent_owner", None)
    server_args = resolved_server_args(config.server_args)
    if owner is None:
        if (
            kvcr_linker_enabled(server_args)
            and getattr(dynamo_args, "use_kv_events", False)
            and not getattr(server_args, "enable_hierarchical_cache", False)
        ):
            # The legacy subscriber would attach to the linker's residency
            # stream, which its normalizer rejects, leaving the indexer blind.
            raise ValueError(
                "The direct KVCR linker with KV events requires "
                "--kvcr-state-agent-owner"
            )
        return None
    if not isinstance(owner, str) or CACHE_OWNER_PATTERN.fullmatch(owner) is None:
        raise ValueError("--kvcr-state-agent-owner requires a canonical CacheOwnerId")
    if config.serving_mode != DisaggregationMode.AGGREGATED:
        raise ValueError("KVCR state agent supports only aggregated LLM workers")
    if not kvcr_linker_enabled(server_args) or not dynamo_args.use_kv_events:
        raise ValueError(
            "KVCR state agent requires the direct KVCR linker and KV events"
        )
    if getattr(server_args, "enable_hierarchical_cache", False):
        raise ValueError(
            "--kvcr-state-agent-owner is for the direct linker; it cannot be "
            "combined with HiCache"
        )
    if (
        getattr(server_args, "nnodes", 1) != 1
        or getattr(server_args, "tp_size", 1) != 1
        or model_card_dp_rank_bounds(server_args) != (0, 1)
        or effective_gateway_workers(server_args, dynamo_args) > 1
    ):
        raise ValueError(
            "KVCR state agent supports only a single-node worker with one TP "
            "rank, one DP rank and one gateway process"
        )
    return owner


def linker_state_agent_descriptor(config: Any, engine: Any) -> dict[str, Any] | None:
    """Read final event and linker coordinates after engine initialization."""
    owner = validate_state_agent_worker(config)
    if owner is None:
        return None
    server_args = resolved_server_args(config.server_args)
    dynamo_args = config.dynamo_args
    events = server_args.kv_events_config
    events = json.loads(events) if isinstance(events, str) else events
    if not isinstance(events, dict) or events.get("publisher") != "zmq":
        raise ValueError("KVCR state agent requires a ZMQ event publisher")
    if events.get("topic", "") != "":
        raise ValueError("KVCR state agent requires an empty event topic")
    endpoint = urlparse(events.get("endpoint", ""))
    if endpoint.scheme != "tcp" or not endpoint.hostname or not endpoint.port:
        raise ValueError("KVCR state agent requires an explicit TCP event port")
    # Registration advertises the same SGLang-reported source; the raw event
    # stream shares its advertise host.
    hint_endpoint = reported_kvcr_hint_endpoints(engine, server_args).get("0")
    if hint_endpoint is None:
        raise ValueError("KVCR state agent requires a linker-reported hint source")
    host = urlparse(hint_endpoint).netloc.rsplit(":", 1)[0]
    validate_advertise_host(host)
    # The linker publishes the legacy residency stream. Name it: the protocol is
    # immutable on a durable host slot, so the native default must not decide it.
    return {
        "cache_owner_id": owner,
        "global_dp_rank": 0,
        "kv_state_endpoint": dynamo_args.kv_state_endpoint
        or f"{dynamo_args.namespace}/{dynamo_args.component}/{dynamo_args.endpoint}",
        "indexer_domain_id": owner.split("/", 1)[0],
        "kv_block_size": kv_event_block_size(server_args),
        "ingress_protocol": "vllm_residency_v1",
        "raw_zmq_endpoint": f"tcp://{host}:{endpoint.port}",
        "raw_topic": "",
        "image_token_id": None,
        "video_token_id": None,
        "router_hint_source": {
            "source_control_endpoint": hint_endpoint,
            "worker_type": "aggregated",
        },
    }


def state_agent_descriptors(engine, config) -> list[dict]:
    sources = inventory_sources(engine)
    if not sources:
        # No HiCache inventory: the direct linker opts in with its own owner.
        descriptor = linker_state_agent_descriptor(config, engine)
        return [] if descriptor is None else [descriptor]
    if getattr(config.dynamo_args, "kvcr_state_agent_owner", None) is not None:
        raise ValueError(
            "--kvcr-state-agent-owner is for the direct linker; KVCR HiCache "
            "inventory reports its own owners"
        )
    server_args = resolved_server_args(config.server_args)
    if (
        not server_args.enable_hierarchical_cache
        or server_args.nnodes != 1
        or server_args.pp_size != 1
        or server_args.tokenizer_worker_num != 1
        or not config.dynamo_args.use_kv_events
    ):
        raise ValueError(
            "KVCR inventory requires a single-host HiCache worker, one tokenizer "
            "worker, PP=1 and KV event emission"
        )
    roles = {
        DisaggregationMode.AGGREGATED: "aggregated",
        DisaggregationMode.PREFILL: "prefill",
        DisaggregationMode.DECODE: "decode",
    }
    role = roles.get(config.serving_mode)
    if role is None:
        raise ValueError("KVCR inventory requires a hint-forwarding LLM worker")
    bounds = model_card_dp_rank_bounds(server_args)
    if set(sources) != set(range(*bounds)):
        raise ValueError("KVCR inventory must cover every managed DP rank")
    infos = engine._scheduler_init_result.scheduler_infos
    endpoints = kvcr_hint_endpoints(infos, bounds)
    if set(endpoints) != {str(rank) for rank in sources}:
        raise ValueError(
            "KVCR inventory requires a router_hint source for every managed DP rank"
        )
    raw_sources: dict[int, Mapping[str, object]] = {}
    for info in infos:
        for source in info.get("kv_event_sources", []):
            rank = source.get("dp_rank")
            if type(rank) is not int or rank not in sources:
                raise ValueError("KVCR raw event source has an invalid DP rank")
            if rank in raw_sources and raw_sources[rank] != source:
                raise ValueError("Conflicting raw KV event sources for one DP rank")
            raw_sources[rank] = source
    if set(raw_sources) != set(sources):
        raise ValueError("KVCR inventory requires a reported raw event source per rank")
    descriptors = []
    owners = set()
    for rank, source in sorted(sources.items()):
        owner = source.get("cache_owner_id")
        if not isinstance(owner, str) or CACHE_OWNER_PATTERN.fullmatch(owner) is None:
            raise ValueError("KVCR inventory source requires a canonical CacheOwnerId")
        if owner in owners:
            raise ValueError("KVCR inventory owners must be unique across DP ranks")
        owners.add(owner)
        raw = raw_sources[rank]
        if raw.get("format") != "dynamo":
            raise ValueError("KVCR inventory requires Dynamo-format KV events")
        endpoint = raw.get("endpoint")
        if not isinstance(endpoint, str):
            raise TypeError("KVCR inventory requires a reported TCP event endpoint")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "tcp"
            or not parsed.hostname
            or parsed.port is None
            or not 0 < parsed.port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or any(char.isspace() for char in parsed.hostname)
            or parsed.hostname == "*"
        ):
            raise ValueError("Invalid reported KVCR raw event endpoint")
        host = parsed.hostname
        validate_advertise_host(f"[{host}]" if ":" in host else host)
        block_size = raw.get("block_size")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("KVCR raw event source requires a positive block size")
        topic = raw.get("topic", "")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(
                "KVCR raw event topic must identify this engine incarnation"
            )
        descriptors.append(
            {
                "cache_owner_id": owner,
                "global_dp_rank": rank,
                "kv_state_endpoint": config.dynamo_args.kv_state_endpoint
                or f"{config.dynamo_args.namespace}/{config.dynamo_args.component}/"
                f"{config.dynamo_args.endpoint}",
                "indexer_domain_id": owner.split("/", 1)[0],
                "kv_block_size": block_size,
                "ingress_protocol": "sglang_hicache_v1",
                "raw_zmq_endpoint": endpoint,
                "raw_topic": topic,
                "image_token_id": None,
                "video_token_id": None,
                "router_hint_source": {
                    "source_control_endpoint": endpoints[str(rank)],
                    "worker_type": role,
                },
            }
        )
    return descriptors


class StateAgentLifecycle:
    def __init__(self):
        self._owner = None
        self._cleanup_task = None

    async def start(self, engine, config, endpoint):
        descriptors = state_agent_descriptors(engine, config)
        if not descriptors:
            return
        owner = KvStateAttachmentOwner(endpoint, endpoint.connection_id(), descriptors)
        startup = asyncio.ensure_future(owner.start())
        try:
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            self._cleanup_task = asyncio.create_task(
                self._close_after_start(owner, startup)
            )
            raise
        self._owner = owner
        logger.info("Attached %d KVCR state-agent owners", len(descriptors))

    async def _close_after_start(self, owner, startup):
        # Let the native binding initialize its OnceCell before withdrawal.
        await startup
        await owner.close()

    async def close(self):
        if self._cleanup_task is not None:
            await bounded_close(asyncio.shield(self._cleanup_task))
            return
        owner, self._owner = self._owner, None
        if owner is not None:
            await bounded_close(owner.close())
