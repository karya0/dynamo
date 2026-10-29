# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang.state_agent import (
    StateAgentLifecycle,
    linker_state_agent_descriptor,
    state_agent_descriptors,
    validate_state_agent_worker,
)

OWNER = f"{'01' * 16}:{'02' * 16}/{'03' * 8}/{'04' * 16}"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _engine(endpoint="tcp://10.0.0.8:19500"):
    sources = (
        [
            {
                "dp_rank": 0,
                "source_control_endpoint": endpoint,
                "capabilities": ["router_hint"],
            }
        ]
        if endpoint
        else []
    )
    return SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(
            scheduler_infos=[{"kv_hint_sources": sources}]
        )
    )


def _config(owner=None):
    return SimpleNamespace(
        serving_mode=DisaggregationMode.AGGREGATED,
        dynamo_args=SimpleNamespace(
            kvcr_state_agent_owner=owner,
            namespace="ns",
            component="backend",
            endpoint="generate",
            kv_state_endpoint=None,
            use_kv_events=True,
        ),
        server_args=SimpleNamespace(
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend="kvcr",
            kv_events_config=json.dumps(
                {"publisher": "zmq", "endpoint": "tcp://*:5557"}
            ),
            tp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            pp_size=1,
            nnodes=1,
            page_size=64,
        ),
    )


@pytest.mark.asyncio
async def test_opt_out_touches_neither_kvcr_nor_the_native_attachment(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("dynamo.sglang.state_agent.KvStateAttachmentOwner", factory)
    config = _config()
    config.server_args = None
    assert linker_state_agent_descriptor(config, None) is None
    lifecycle = StateAgentLifecycle()
    await lifecycle.start(None, config, Mock())
    await lifecycle.close()
    factory.assert_not_called()


def test_descriptor_preserves_owner_and_exact_source_coordinates():
    # These ports are protocol fixtures; no sockets are opened.
    assert linker_state_agent_descriptor(_config(OWNER), _engine()) == {
        "cache_owner_id": OWNER,
        "global_dp_rank": 0,
        "kv_state_endpoint": "ns/backend/generate",
        "indexer_domain_id": OWNER.split("/", 1)[0],
        "kv_block_size": 64,
        "ingress_protocol": "vllm_residency_v1",
        "raw_zmq_endpoint": "tcp://10.0.0.8:5557",
        "raw_topic": "",
        "image_token_id": None,
        "video_token_id": None,
        "router_hint_source": {
            "source_control_endpoint": "tcp://10.0.0.8:19500",
            "worker_type": "aggregated",
        },
    }


_TOPIC = '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"other"}'


@pytest.mark.parametrize(
    "target,field,value,match",
    [
        ("config", "serving_mode", DisaggregationMode.DECODE, "aggregated"),
        ("server", "enable_unified_cache_external_linker", False, "linker"),
        ("dynamo", "use_kv_events", False, "KV events"),
        ("server", "nnodes", 2, "single-node"),
        ("server", "attn_dp_size", 2, "single-node"),
        ("server", "tp_size", 2, "single-node"),
        ("server", "enable_hierarchical_cache", True, "HiCache"),
        ("dynamo", "gateway_workers", 2, "single-node"),
        *[
            (
                "server",
                "kv_events_config",
                json.dumps({"publisher": "zmq", "endpoint": e}),
                "event",
            )
            for e in (
                "ipc:///tmp/events",
                "tcp://*:0",
            )
        ],
        ("server", "kv_events_config", _TOPIC, "topic"),
        *[
            ("dynamo", "kvcr_state_agent_owner", o, "CacheOwnerId")
            for o in ("", "not-an-owner", OWNER.replace("01", "AA"))
        ],
    ],
)
def test_rejects_invalid_configuration(target, field, value, match):
    config = _config(OWNER)
    holder = {
        "config": config,
        "server": config.server_args,
        "dynamo": config.dynamo_args,
    }
    setattr(holder[target], field, value)
    with pytest.raises(ValueError, match=match):
        linker_state_agent_descriptor(config, _engine())


def test_rejects_unsafe_hint_host_and_linker_without_owner():
    with pytest.raises(ValueError, match="raw_advertise_host"):
        linker_state_agent_descriptor(_config(OWNER), _engine("tcp://localhost:19500"))
    with pytest.raises(ValueError, match="requires --kvcr-state-agent-owner"):
        validate_state_agent_worker(_config(None))
    config = _config(None)
    config.dynamo_args.use_kv_events = False
    assert validate_state_agent_worker(config) is None


def test_descriptors_route_linker_opt_in_and_reject_mixing_with_hicache_inventory():
    legacy = _config(None)
    legacy.dynamo_args.use_kv_events = False
    assert state_agent_descriptors(_engine(), legacy) == []
    assert state_agent_descriptors(_engine(), _config(OWNER)) == [
        linker_state_agent_descriptor(_config(OWNER), _engine())
    ]
    engine = _engine()
    engine._scheduler_init_result.scheduler_infos[0]["kv_hint_sources"][0][
        "capabilities"
    ] = ["router_hint", "kvcr_inventory"]
    with pytest.raises(ValueError, match="direct linker"):
        state_agent_descriptors(engine, _config(OWNER))


def test_requires_a_linker_reported_hint_source():
    with pytest.raises(ValueError, match="linker-reported"):
        linker_state_agent_descriptor(_config(OWNER), _engine(None))
    with pytest.raises(ValueError, match="linker-reported"):
        linker_state_agent_descriptor(_config(OWNER), SimpleNamespace())


@pytest.mark.asyncio
async def test_lifecycle_starts_one_owner_and_closes_once(monkeypatch):
    owner = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    factory = Mock(return_value=owner)
    monkeypatch.setattr("dynamo.sglang.state_agent.KvStateAttachmentOwner", factory)
    endpoint = Mock()
    endpoint.connection_id.return_value = 123
    lifecycle = StateAgentLifecycle()
    await lifecycle.start(_engine(), _config(OWNER), endpoint)
    factory.assert_called_once_with(
        endpoint, 123, [linker_state_agent_descriptor(_config(OWNER), _engine())]
    )
    owner.start.assert_awaited_once()
    await lifecycle.close()
    await lifecycle.close()
    owner.close.assert_awaited_once()


def test_config_rejects_bad_owner_before_engine_startup():
    from dynamo.sglang.args import Config

    config = _config("invalid")
    config.server_args.disaggregation_mode = "null"
    with pytest.raises(ValueError, match="CacheOwnerId"):
        Config(config.server_args, config.dynamo_args)


def test_early_launch_validation_does_not_require_final_page_size():
    from sglang.srt.server_args import ServerArgs

    from dynamo.sglang.args import Config

    config = _config(OWNER)
    raw = ServerArgs(
        model_path="dummy",
        page_size=None,
        attn_cp_size=None,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        kv_events_config=config.server_args.kv_events_config,
    )
    assert Config(raw, config.dynamo_args).server_args is raw


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, OWNER])
async def test_registration_advertises_state_agent_only_when_opted_in(owner):
    from sglang.srt.server_args import ServerArgs

    from dynamo.sglang.register import DynamoConfig, get_runtime_config

    args = DynamoConfig()
    args.kvcr_state_agent_owner = owner
    args.use_kv_events = True
    args.enable_local_indexer = False
    runtime_config = await get_runtime_config(
        None, ServerArgs(model_path="dummy", page_size=64), args
    )
    assert runtime_config.kv_event_publishing_enabled is True
    assert runtime_config.kv_event_source_mode == ("state_agent_v2" if owner else None)


@pytest.mark.parametrize("opt_in", [False, True])
def test_only_opt_in_suppresses_legacy_publisher(opt_in, monkeypatch):
    import dynamo.sglang.publisher as module

    config = _config(OWNER if opt_in else None)
    config.dynamo_args.enable_local_indexer = False
    publisher = module.DynamoSglangPublisher.__new__(module.DynamoSglangPublisher)
    publisher.engine = (
        _engine()
    )  # no kvcr_inventory source: HiCache inventory path stays off
    publisher.dynamo_args = config.dynamo_args
    publisher.server_args = config.server_args
    publisher.generate_endpoint = Mock()
    publisher.kv_worker_id = 123
    publisher.kv_publishers = []
    factory = Mock()
    monkeypatch.setattr(module, "KvEventPublisher", factory)
    assert len(publisher.init_kv_event_publish()) == (0 if opt_in else 1)
    assert factory.call_count == (0 if opt_in else 1)
    assert config.dynamo_args.use_kv_events is True
