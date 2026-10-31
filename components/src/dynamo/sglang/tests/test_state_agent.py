# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common import kv_state_agent
from dynamo.common.constants import (
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    DisaggregationMode,
)
from dynamo.llm import HttpError, WorkerType

pytest.importorskip("sglang")

from dynamo.sglang import state_agent  # noqa: E402
from dynamo.sglang.state_agent import (  # noqa: E402
    StateAgentLifecycle,
    state_agent_descriptors,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

LINKER_WORKER_MODES = (
    DisaggregationMode.AGGREGATED,
    DisaggregationMode.PREFILL,
    DisaggregationMode.DECODE,
)


def fixture():
    source = {
        "dp_rank": 0,
        "capabilities": ["router_hint", "kvcr_inventory"],
        "source_control_endpoint": "tcp://worker:24100",
        "cache_owner_id": f"{'01' * 16}:{'02' * 16}/{'00' * 8}/{'03' * 16}",
    }
    info = {
        "kv_hint_sources": [source],
        "kv_event_sources": [
            {
                "dp_rank": 0,
                "endpoint": "tcp://worker:24020",
                "topic": "raw",
                "block_size": 64,
                "format": "dynamo",
            }
        ],
    }
    engine = SimpleNamespace(
        _scheduler_init_result=SimpleNamespace(scheduler_infos=[info])
    )
    config = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_hierarchical_cache=True,
            dp_size=1,
            nnodes=1,
            pp_size=1,
            tokenizer_worker_num=1,
        ),
        dynamo_args=SimpleNamespace(
            use_kv_events=True,
            kv_state_endpoint=None,
            namespace="ns",
            component="backend",
            endpoint="generate",
        ),
        serving_mode=DisaggregationMode.AGGREGATED,
    )
    return engine, config, source, info


def test_descriptors_use_only_reported_sources():
    engine, config, source, _ = fixture()
    (result,) = state_agent_descriptors(engine, config)
    assert result["cache_owner_id"] == source["cache_owner_id"]
    assert (
        result["router_hint_source"]["source_control_endpoint"] == "tcp://worker:24100"
    )
    assert result["raw_zmq_endpoint"] == "tcp://worker:24020"
    assert result["ingress_protocol"] == "sglang_hicache_v1"
    assert result["kv_state_endpoint"] == "ns/backend/generate"
    source["capabilities"] = ["router_hint"]
    assert state_agent_descriptors(engine, config) == []


def linker_fixture():
    engine, config, source, info = fixture()
    config.server_args.enable_hierarchical_cache = False
    config.server_args.enable_unified_cache_external_linker = True
    config.server_args.unified_cache_external_linker_backend = "kvcr"
    config.server_args.tp_size = 1
    config.server_args.page_size = 64
    source["inventory_lifetime"] = "process_local"
    source["reset_clears_owner_inventory"] = True
    source["owner_incarnation_scoped"] = True
    info["kv_event_sources"][0]["topic"] = kv_state_agent.worker_raw_topic(
        [source["cache_owner_id"]]
    )

    async def async_generate(*, kv_hints=None):
        return kv_hints

    engine.async_generate = async_generate
    return engine, config, source, info


@pytest.mark.parametrize("mode", LINKER_WORKER_MODES)
def test_linker_descriptor_uses_reported_owner_endpoint_and_event_source(mode):
    engine, config, source, _ = linker_fixture()
    config.serving_mode = mode
    (descriptor,) = state_agent_descriptors(engine, config)
    assert descriptor["cache_owner_id"] == source["cache_owner_id"]
    assert descriptor["raw_zmq_endpoint"] == "tcp://worker:24020"
    assert descriptor["raw_topic"] == kv_state_agent.worker_raw_topic(
        [source["cache_owner_id"]]
    )
    assert descriptor["kv_block_size"] == 64
    assert descriptor["router_hint_source"] == {
        "source_control_endpoint": "tcp://worker:24100",
        "worker_type": {
            DisaggregationMode.AGGREGATED: "aggregated",
            DisaggregationMode.PREFILL: "prefill",
            DisaggregationMode.DECODE: "decode",
        }[mode],
    }


def guard_fixture():
    engine, config, source, info = linker_fixture()
    owner = source["cache_owner_id"]
    source.clear()
    source.update(
        {
            "dp_rank": 0,
            "capabilities": ["router_hint"],
            "source_control_endpoint": "tcp://10.0.20.27:19500",
            "inventory_lifetime": "guard",
            "reset_clears_owner_inventory": False,
            "owner_incarnation_scoped": False,
        }
    )
    del info["kv_event_sources"]
    config.server_args.kv_events_config = json.dumps(
        {"publisher": "zmq", "endpoint": "tcp://0.0.0.0:24020", "topic": ""}
    )
    config.dynamo_args.kvcr_state_agent_owner = owner
    return engine, config, source, info


def test_guard_source_yields_vllm_residency_descriptor():
    engine, config, source, _ = guard_fixture()
    (descriptor,) = state_agent_descriptors(engine, config)
    assert descriptor["cache_owner_id"] == config.dynamo_args.kvcr_state_agent_owner
    assert descriptor["global_dp_rank"] == 0
    assert descriptor["kv_block_size"] == 64
    assert descriptor["ingress_protocol"] == "vllm_residency_v1"
    assert descriptor["raw_zmq_endpoint"] == "tcp://10.0.20.27:24020"
    assert descriptor["raw_topic"] == ""
    assert descriptor["router_hint_source"] == {
        "source_control_endpoint": source["source_control_endpoint"],
        "worker_type": "aggregated",
    }


def test_mixed_sources_raise():
    engine, config, _, info = guard_fixture()
    _, _, inventory, _ = linker_fixture()
    info["kv_hint_sources"].append(inventory)
    with pytest.raises(ValueError, match="cannot be mixed"):
        state_agent_descriptors(engine, config)


def test_guard_without_owner_id_raises():
    engine, config, _, _ = guard_fixture()
    config.dynamo_args.kvcr_state_agent_owner = None
    with pytest.raises(ValueError, match="kvcr-state-agent-owner"):
        state_agent_descriptors(engine, config)
    config.dynamo_args.kvcr_state_agent_owner = "not-an-owner"
    with pytest.raises(ValueError, match="kvcr-state-agent-owner"):
        state_agent_descriptors(engine, config)


def test_linker_registration_uses_state_agent(monkeypatch):
    from dynamo.sglang import register

    engine, config, source, _ = linker_fixture()
    config.server_args.model_path = config.server_args.served_model_name = "test-model"
    config.dynamo_args.use_sglang_tokenizer = False
    config.dynamo_args.model_source_uri = None
    config.dynamo_args.custom_jinja_template = None
    values = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: values.update({key: json.loads(value)})
    )
    monkeypatch.setattr(register, "get_runtime_config", AsyncMock(return_value=runtime))
    monkeypatch.setattr(register, "register_model", AsyncMock())
    assert list(register.inventory_sources(engine).values()) == [source]
    asyncio.run(
        register.register_model_with_readiness_gate(
            engine,
            None,
            config.server_args,
            config.dynamo_args,
            worker_type=WorkerType.Prefill,
            forwards_kv_hints=True,
        )
    )
    assert runtime.kv_event_source_mode == "state_agent_v2"
    assert values[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY] == {
        "0": source["source_control_endpoint"]
    }


def test_guard_registration_uses_state_agent(monkeypatch):
    from dynamo.sglang import register

    engine, config, source, _ = guard_fixture()
    config.server_args.model_path = config.server_args.served_model_name = "test-model"
    config.dynamo_args.use_sglang_tokenizer = False
    config.dynamo_args.model_source_uri = None
    config.dynamo_args.custom_jinja_template = None
    values = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: values.update({key: json.loads(value)})
    )
    monkeypatch.setattr(register, "get_runtime_config", AsyncMock(return_value=runtime))
    monkeypatch.setattr(register, "register_model", AsyncMock())
    assert register.inventory_sources(engine) == {}
    asyncio.run(
        register.register_model_with_readiness_gate(
            engine,
            None,
            config.server_args,
            config.dynamo_args,
            worker_type=WorkerType.Aggregated,
            forwards_kv_hints=True,
        )
    )
    assert runtime.kv_event_source_mode == "state_agent_v2"
    assert values[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY] == {
        "0": source["source_control_endpoint"]
    }


@pytest.mark.parametrize(
    "change,mode,match",
    [
        ({"tp_size": 2}, DisaggregationMode.AGGREGATED, "supported linker"),
        ({}, DisaggregationMode.ENCODE, "hint-forwarding LLM worker"),
    ],
)
def test_linker_inventory_rejects_unsupported_layouts(change, mode, match):
    engine, config, _, _ = linker_fixture()
    vars(config.server_args).update(change)
    config.serving_mode = mode
    with pytest.raises(ValueError, match=match):
        state_agent_descriptors(engine, config)


def test_linker_inventory_requires_hint_forwarding_and_reported_inventory():
    engine, config, source, info = linker_fixture()
    config.serving_mode = DisaggregationMode.PREFILL
    engine.async_generate = lambda: None
    with pytest.raises(ValueError, match="hint-forwarding"):
        state_agent_descriptors(engine, config)
    source["capabilities"] = ["router_hint"]
    assert state_agent_descriptors(engine, config) == []
    del info["kv_hint_sources"]
    assert state_agent_descriptors(engine, config) == []


@pytest.mark.parametrize(
    "target,field,value,match",
    [
        ("source", "inventory_lifetime", "persistent", "process-local"),
        ("source", "reset_clears_owner_inventory", False, "reset and incarnation"),
        ("source", "owner_incarnation_scoped", False, "reset and incarnation"),
        ("event", "block_size", 128, "block size"),
        ("event", "topic", "other", "topic must identify"),
    ],
)
def test_linker_inventory_rejects_wrong_ownership_contract(target, field, value, match):
    engine, config, source, info = linker_fixture()
    config.serving_mode = DisaggregationMode.PREFILL
    {"source": source, "event": info["kv_event_sources"][0]}[target][field] = value
    with pytest.raises(ValueError, match=match):
        state_agent_descriptors(engine, config)


def test_linker_event_block_size_matches_router_block_size(monkeypatch):
    engine, config, _, info = linker_fixture()
    info["kv_event_sources"][0]["block_size"] = 128
    monkeypatch.setattr(state_agent, "kv_event_block_size", lambda args: 128)
    (descriptor,) = state_agent_descriptors(engine, config)
    assert descriptor["kv_block_size"] == 128


@pytest.mark.asyncio
async def test_linker_invalid_inventory_fails_before_attachment(monkeypatch):
    engine, config, source, _ = linker_fixture()
    source["reset_clears_owner_inventory"] = False
    owner = Mock()
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", owner)
    with pytest.raises(ValueError, match="reset and incarnation"):
        await StateAgentLifecycle().start(engine, config, None)
    owner.assert_not_called()


@pytest.mark.parametrize(
    "target,updates,match",
    [
        ("info", {"kv_event_sources": []}, "raw event source per rank"),
        ("raw", {"endpoint": "tcp://0.0.0.0:24020"}, "non-loopback"),
        ("server_args", {"dp_size": 2}, "every managed DP rank"),
        (
            "source",
            {"capabilities": ["kvcr_inventory"]},
            "router_hint source for every managed DP rank",
        ),
        ("dynamo_args", {"use_kv_events": False}, "KV event emission"),
        ("raw", {"format": "default"}, "Dynamo-format"),
        ("raw", {"topic": ""}, "engine incarnation"),
    ],
)
def test_inventory_fails_closed_for_incomplete_startup_metadata(target, updates, match):
    engine, config, source, info = fixture()
    targets = {
        "info": info,
        "source": source,
        "raw": info["kv_event_sources"][0],
        "server_args": vars(config.server_args),
        "dynamo_args": vars(config.dynamo_args),
    }
    targets[target].update(updates)
    with pytest.raises(ValueError, match=match):
        state_agent_descriptors(engine, config)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "127.1",
        "2130706433",
        "0",
        "[::1]",
        "[::ffff:127.0.0.1]",
    ],
)
def test_raw_event_endpoint_rejects_loopback_and_unspecified_hosts(host):
    engine, config, _, info = fixture()
    info["kv_event_sources"][0]["endpoint"] = f"tcp://{host}:24020"
    with pytest.raises(ValueError, match="non-loopback"):
        state_agent_descriptors(engine, config)


def test_raw_event_endpoint_accepts_connectable_ipv6():
    engine, config, _, info = fixture()
    endpoint = "tcp://[2001:db8::1]:24020"
    info["kv_event_sources"][0]["endpoint"] = endpoint
    (descriptor,) = state_agent_descriptors(engine, config)
    assert descriptor["raw_zmq_endpoint"] == endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_cleanup_timeout_preserves_body_exception(
    monkeypatch, caplog, error_type
):
    engine, config, _, _ = fixture()
    owner = SimpleNamespace(
        start=AsyncMock(),
        await_attached=AsyncMock(),
        close=AsyncMock(side_effect=asyncio.Event().wait),
    )
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
    monkeypatch.setattr(kv_state_agent, "_STATE_AGENT_CLOSE_TIMEOUT_SECS", 0.01)
    lifecycle = StateAgentLifecycle()
    await lifecycle.start(engine, config, SimpleNamespace(connection_id=lambda: 17))
    original = error_type("worker body failed")

    with pytest.raises(error_type) as caught:
        try:
            raise original
        finally:
            await lifecycle.close()

    assert caught.value is original
    assert "cleanup timed out" in caplog.text
    await lifecycle.close()
    owner.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_waits_for_attach_ack(monkeypatch):
    engine, config, _, _ = fixture()
    ack = asyncio.Event()

    async def await_attached():
        await ack.wait()

    owner = SimpleNamespace(
        start=AsyncMock(), await_attached=await_attached, close=AsyncMock()
    )
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
    lifecycle = StateAgentLifecycle()
    endpoint = SimpleNamespace(connection_id=lambda: 17)
    await lifecycle.start(engine, config, endpoint)
    owner.start.assert_awaited_once()
    assert lifecycle.active
    task = asyncio.create_task(lifecycle.await_attached())
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not lifecycle.attached
    ack.set()
    await asyncio.wait_for(task, timeout=1)
    assert lifecycle.attached
    await lifecycle.close()
    owner.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ack_timeout_closes_retained_owner(monkeypatch):
    engine, config, _, _ = fixture()
    owner = SimpleNamespace(
        start=AsyncMock(),
        await_attached=AsyncMock(side_effect=RuntimeError("did not acknowledge")),
        close=AsyncMock(),
    )
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
    lifecycle = StateAgentLifecycle()
    await lifecycle.start(engine, config, SimpleNamespace(connection_id=lambda: 17))
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        await lifecycle.await_attached()
    assert not lifecycle.attached
    assert lifecycle._owner is owner
    await lifecycle.close()
    owner.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_start_cleanup_timeout_preserves_withdrawal(
    monkeypatch, caplog
):
    engine, config, _, _ = fixture()
    started = asyncio.Event()
    native_result = asyncio.get_running_loop().create_future()

    def start():
        started.set()
        return native_result

    async def close():
        assert native_result.done() and not native_result.cancelled()

    owner = SimpleNamespace(
        start=start, await_attached=AsyncMock(), close=AsyncMock(side_effect=close)
    )
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
    monkeypatch.setattr(kv_state_agent, "_STATE_AGENT_CLOSE_TIMEOUT_SECS", 0.01)
    lifecycle = StateAgentLifecycle()
    task = asyncio.create_task(
        lifecycle.start(engine, config, SimpleNamespace(connection_id=lambda: 17))
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        try:
            await task
        finally:
            await lifecycle.close()

    assert "cleanup timed out" in caplog.text
    assert not native_result.done()
    owner.close.assert_not_awaited()
    native_result.set_result(None)
    await asyncio.wait_for(lifecycle.close(), timeout=1)
    owner.close.assert_awaited_once()
    assert lifecycle._owner is None


@pytest.mark.parametrize("native_finished_first", [False, True])
def test_cancelled_start_withdraws_registered_owner(monkeypatch, native_finished_first):
    async def exercise():
        engine, config, _, _ = fixture()
        started = asyncio.Event()
        native_result = asyncio.get_running_loop().create_future()

        def start():
            started.set()
            return native_result

        async def close():
            assert native_result.done() and not native_result.cancelled()

        owner = SimpleNamespace(
            start=start, await_attached=AsyncMock(), close=AsyncMock(side_effect=close)
        )
        monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
        lifecycle = StateAgentLifecycle()
        endpoint = SimpleNamespace(connection_id=lambda: 17)
        task = asyncio.create_task(lifecycle.start(engine, config, endpoint))
        await asyncio.wait_for(started.wait(), timeout=1)
        if native_finished_first:
            native_result.set_result(None)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not native_result.cancelled()
        if not native_finished_first:
            owner.close.assert_not_awaited()
            native_result.set_result(None)
        await asyncio.wait_for(lifecycle.close(), timeout=1)
        owner.close.assert_awaited_once()
        assert lifecycle._owner is None

    asyncio.run(exercise())


class _Lifecycle:
    def __init__(self, events, active=True, error=None):
        self.events, self.active, self.error = events, active, error

    async def await_attached(self):
        self.events.append("await_attached")
        await asyncio.sleep(0)
        if self.error:
            raise self.error


def _serve_and_register(events):
    async def serve():
        events.append("serve")
        await asyncio.sleep(0)

    async def register():
        events.append("register")

    return [serve(), serve()], register


@pytest.mark.asyncio
async def test_serve_starts_before_attach_and_register_follows_attach():
    # Helper-level only: init_decode/init_prefill call it, but their full
    # init order is not exercised here.
    from dynamo.sglang.init_llm import _serve_attach_register

    events = []
    serve_tasks, register = _serve_and_register(events)
    await _serve_attach_register(_Lifecycle(events), serve_tasks, register)
    assert events == ["serve", "serve", "await_attached", "register"]


@pytest.mark.asyncio
async def test_non_guard_worker_registers_without_attach():
    from dynamo.sglang.init_llm import _serve_attach_register

    events = []
    serve_tasks, register = _serve_and_register(events)
    lifecycle = _Lifecycle(events, active=False)
    await _serve_attach_register(lifecycle, serve_tasks, register)
    assert "await_attached" not in events
    assert events.count("register") == 1


@pytest.mark.asyncio
async def test_attach_failure_fails_worker_without_registering_and_withdraws(
    monkeypatch,
):
    from dynamo.sglang.init_llm import _serve_attach_register

    engine, config, _, _ = fixture()
    owner = SimpleNamespace(
        start=AsyncMock(),
        await_attached=AsyncMock(side_effect=RuntimeError("did not acknowledge")),
        close=AsyncMock(),
    )
    monkeypatch.setattr(state_agent, "KvStateAttachmentOwner", lambda *args: owner)
    lifecycle = StateAgentLifecycle()
    await lifecycle.start(engine, config, SimpleNamespace(connection_id=lambda: 17))
    events = []
    serve_tasks, register = _serve_and_register(events)
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        await _serve_attach_register(lifecycle, serve_tasks, register)
    assert "register" not in events
    await lifecycle.close()
    owner.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["decode", "prefill"])
async def test_handler_rejects_until_attached(role):
    from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
    from dynamo.sglang.request_handlers.llm.prefill_handler import (
        PrefillWorkerHandler,
    )

    cls = DecodeWorkerHandler if role == "decode" else PrefillWorkerHandler
    handler = cls.__new__(cls)
    # No lifecycle (non-Guard): never rejects.
    handler._raise_if_not_attached()
    handler.attach_gate = SimpleNamespace(attached=False)
    with pytest.raises(HttpError):
        async for _ in handler.generate({}, Mock()):
            pass
    handler.attach_gate.attached = True
    handler._raise_if_not_attached()


@pytest.mark.parametrize(
    "active,enabled,kept",
    [(True, "true", False), (True, "false", True), (False, "true", True)],
)
def test_guard_canary_dropped_only_with_guard_and_health_check_enabled(
    monkeypatch, active, enabled, kept
):
    from dynamo.sglang.init_llm import _guard_health_check_payload

    monkeypatch.setenv("DYN_HEALTH_CHECK_ENABLED", enabled)
    payload = {"request": 1}
    result = _guard_health_check_payload(SimpleNamespace(active=active), payload)
    assert (result is payload) if kept else (result is None)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["load_lora", "unload_lora", "clear_kv_blocks"])
async def test_mutating_control_methods_gated_until_attached(method):
    from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.attach_gate = SimpleNamespace(attached=False)
    with pytest.raises(HttpError):
        async for _ in getattr(handler, method)({}):
            pass
    handler.attach_gate.attached = True
    try:
        async for _ in getattr(handler, method)({}):
            pass
    except HttpError:
        pytest.fail("gate must pass once attached")
    except Exception:  # noqa: BLE001 - bare handler; only the gate matters
        pass


@pytest.mark.asyncio
async def test_engine_route_rejected_until_attached():
    from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.attach_gate = SimpleNamespace(attached=False)
    route = AsyncMock(return_value={"ok": True})
    wrapped = handler._wrap_engine_route(route)
    with pytest.raises(HttpError):
        await wrapped({})
    route.assert_not_awaited()
