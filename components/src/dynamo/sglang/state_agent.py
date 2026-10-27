# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in state-agent attachment for an aggregated KVCR linker worker."""

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import KvStateAttachmentOwner
from dynamo.sglang._compat import resolved_server_args
from dynamo.sglang.capacity import kv_event_block_size
from dynamo.sglang.kv_hints import kvcr_linker_enabled, reported_kvcr_hint_endpoints

_CACHE_OWNER_PATTERN = re.compile(
    r"^[0-9a-f]{32}:[0-9a-f]{32}/[0-9a-f]{16}/[0-9a-f]{32}$"
)
_CLEANUP_TIMEOUT_SECS = 30.0


class StateAgentLifecycle:
    """Start one attachment after engine init and close it once on shutdown."""

    def __init__(self) -> None:
        self._owner: KvStateAttachmentOwner | None = None

    async def start(self, config: Any, endpoint: Any, engine: Any) -> None:
        descriptor = state_agent_descriptor(config, engine)
        if descriptor is None:
            return
        owner = KvStateAttachmentOwner(endpoint, endpoint.connection_id(), [descriptor])
        # Cancelling this await drops the native waiter, which withdraws its
        # own registrations; nothing is left for close() to undo.
        await owner.start()
        self._owner = owner

    async def close(self) -> None:
        owner, self._owner = self._owner, None
        if owner is None:
            return
        try:
            await asyncio.wait_for(owner.close(), _CLEANUP_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            logging.warning(
                "KV state attachment cleanup timed out; native cleanup continues"
            )


def validate_state_agent_worker(config: Any) -> str | None:
    """Validate launch intent without forcing SGLang's engine resolution."""
    dynamo_args = config.dynamo_args
    owner = getattr(dynamo_args, "kvcr_state_agent_owner", None)
    if owner is None:
        return None
    if not isinstance(owner, str) or not _CACHE_OWNER_PATTERN.fullmatch(owner):
        raise ValueError("--kvcr-state-agent-owner requires a canonical CacheOwnerId")
    server_args = resolved_server_args(config.server_args)
    if config.serving_mode != DisaggregationMode.AGGREGATED:
        raise ValueError("KVCR state agent supports only aggregated LLM workers")
    if not kvcr_linker_enabled(server_args) or not dynamo_args.use_kv_events:
        raise ValueError(
            "KVCR state agent requires the direct KVCR linker and KV events"
        )
    return owner


def state_agent_descriptor(config: Any, engine: Any) -> dict[str, Any] | None:
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
    return {
        "cache_owner_id": owner,
        "global_dp_rank": 0,
        "kv_state_endpoint": dynamo_args.kv_state_endpoint
        or f"{dynamo_args.namespace}/{dynamo_args.component}/{dynamo_args.endpoint}",
        "indexer_domain_id": owner.split("/", 1)[0],
        "kv_block_size": kv_event_block_size(server_args),
        "raw_zmq_endpoint": f"tcp://{host}:{endpoint.port}",
        "raw_topic": "",
        "image_token_id": None,
        "video_token_id": None,
        "router_hint_source": {
            "source_control_endpoint": hint_endpoint,
            "worker_type": "aggregated",
        },
    }
