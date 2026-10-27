# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV-hint forwarding and KVCR source registration."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import ModelType, WorkerType
from dynamo.sglang import register
from dynamo.sglang.kv_hints import (
    kv_hint_kwargs,
    kvcr_hint_endpoints,
    kvcr_mode,
    publish_kvcr_hint_capabilities,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.router,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


class _HintEngine:
    async def async_generate(self, *, kv_hints=None):
        return kv_hints


class _OldEngine:
    async def async_generate(self, *, input_ids):
        return input_ids


def test_router_hint_forwarding_is_private_and_lossless():
    hint = {
        "protocol_version": "0.1",
        "message_id": "selected-source",
        "actions": [
            {"action_type": "kv.fetch", "payload": {"block_hashes": [(1 << 63) + 1]}}
        ],
    }
    assert kv_hint_kwargs(_HintEngine(), {"kv_hint": hint}) == {"kv_hints": hint}
    assert kv_hint_kwargs(_OldEngine(), {"kv_hint": hint}) == {}
    assert kv_hint_kwargs(_HintEngine(), {"extra_args": {"kv_hint": hint}}) == {}


def test_endpoint_layout_matches_attention_group_offsets():
    control_port = 28000
    result = kvcr_hint_endpoints(
        extra_config={
            "control_advertise_host": "127.0.0.1",
            "control_port": control_port,
        },
        dp_size=2,
        tp_size=2,
        attn_cp_size=2,
        pp_size=1,
    )
    assert result == {
        "0": f"tcp://127.0.0.1:{control_port}",
        "1": f"tcp://127.0.0.1:{control_port + 4}",
    }


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"nnodes": 2},
        {"pp_size": 2},
        {"hicache_storage_backend": "mooncake"},
        {"enable_hierarchical_cache": False},
    ],
)
def test_registration_only_advertises_supported_transfer_path(change):
    control_port = 28000
    args = SimpleNamespace(
        enable_hierarchical_cache=True,
        hicache_storage_backend="kvcr",
        nnodes=1,
        attn_dp_size=2,
        attn_cp_size=1,
        tp_size=4,
        pp_size=1,
        hicache_storage_backend_extra_config=json.dumps(
            {"control_port": control_port, "control_advertise_host": "127.0.0.1"}
        ),
    )
    for name, value in change.items():
        setattr(args, name, value)
    metadata = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: metadata.__setitem__(
            key, json.loads(value)
        )
    )
    publish_kvcr_hint_capabilities(runtime, _HintEngine(), args, WorkerType.Aggregated)
    if change:
        assert not metadata
    else:
        assert metadata[KV_HINT_TRANSFER_CAPABILITY_KEY] is True
        assert metadata[KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY] == "aggregated"
        assert (
            metadata[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY]["1"]
            == f"tcp://127.0.0.1:{control_port + 2}"
        )
    metadata.clear()
    publish_kvcr_hint_capabilities(runtime, _OldEngine(), args, WorkerType.Aggregated)
    assert not metadata


@pytest.mark.parametrize(
    "extra",
    [
        {"control_advertise_host": "0.0.0.0"},
        {"control_port": True},
        {"control_port": 65535},
    ],
)
def test_invalid_advertisement_fails_closed(extra):
    with pytest.raises(ValueError):
        kvcr_hint_endpoints(
            extra_config=extra, dp_size=2, tp_size=4, attn_cp_size=1, pp_size=1
        )


def _linker_registration(extra, **changes):
    args = SimpleNamespace(
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="kvcr",
        nnodes=1,
        attn_dp_size=2,
        attn_cp_size=1,
        tp_size=4,
        pp_size=1,
        hicache_storage_backend_extra_config=extra,
    )
    for name, value in changes.items():
        setattr(args, name, value)
    metadata = {}
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: metadata.__setitem__(
            key, json.loads(value)
        )
    )
    publish_kvcr_hint_capabilities(runtime, _HintEngine(), args, WorkerType.Decode)
    return args, metadata


@pytest.mark.parametrize("host_key", ["advertise_host", "control_advertise_host"])
def test_linker_registration_uses_explicit_backend_coordinates(host_key):
    extra = {host_key: "192.0.2.1", "control_port": 29000}
    if host_key == "control_advertise_host":
        extra["enable_remote_hint"] = True
    args, metadata = _linker_registration(extra)
    assert kvcr_mode(args) == "linker"
    assert metadata[KV_HINT_TRANSFER_CAPABILITY_KEY] is True
    assert metadata[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY] == {
        "0": "tcp://192.0.2.1:29000",
        "1": "tcp://192.0.2.1:29002",
    }


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"control_port": 29000},
        {"advertise_host": "192.0.2.1"},
        {"control_advertise_host": "192.0.2.1", "control_port": 29000},
        {
            "advertise_host": "192.0.2.1",
            "control_port": 29000,
            "enable_remote_hint": False,
        },
    ],
)
def test_local_only_linker_does_not_publish_guessed_sources(extra):
    _, metadata = _linker_registration(extra)
    assert not metadata


def test_hicache_does_not_consume_the_linker_only_host_alias():
    endpoints = kvcr_hint_endpoints(
        extra_config={"advertise_host": "192.0.2.1", "control_port": 29000},
        dp_size=1,
        tp_size=1,
        attn_cp_size=1,
        pp_size=1,
    )
    assert endpoints == {"0": "tcp://127.0.0.1:29000"}


@pytest.mark.parametrize(
    "extra",
    [
        {"advertise_host": "192.0.2.1", "control_advertise_host": "192.0.2.2"},
        {"advertise_host": "0.0.0.0"},
        {"advertise_host": "tcp://192.0.2.1"},
        {"advertise_host": "192.0.2.1", "control_port": 0},
    ],
)
def test_linker_invalid_coordinates_fail_closed(extra):
    with pytest.raises(ValueError):
        _linker_registration(
            {"advertise_host": "192.0.2.1", "control_port": 29000, **extra}
        )


def test_linker_ipv6_group_offsets_are_unambiguous():
    _, metadata = _linker_registration(
        {
            "control_advertise_host": "2001:db8::1",
            "control_port": 29000,
            "enable_remote_hint": True,
        }
    )
    assert metadata[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY] == {
        "0": "tcp://[2001:db8::1]:29000",
        "1": "tcp://[2001:db8::1]:29002",
    }


@pytest.mark.parametrize("mode", ["hicache", "linker"])
def test_registration_reads_at_file_config(tmp_path, mode):
    path = tmp_path / "extra.json"
    path.write_text(
        json.dumps(
            {
                "control_advertise_host": "192.0.2.1",
                "control_port": 29000,
                "enable_remote_hint": True,
            }
        ),
        encoding="utf-8",
    )
    _, metadata = _linker_registration(
        f"  @{path}  ",
        enable_unified_cache_external_linker=mode == "linker",
        enable_hierarchical_cache=mode == "hicache",
        hicache_storage_backend="kvcr",
    )
    assert metadata[KV_HINT_TRANSFER_CAPABILITY_KEY] is True
    assert metadata[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY] == {
        "0": "tcp://192.0.2.1:29000",
        "1": "tcp://192.0.2.1:29002",
    }


def test_registration_normalizes_blank_config():
    extra = " \n\t "
    _, metadata = _linker_registration(
        extra,
        enable_unified_cache_external_linker=False,
        enable_hierarchical_cache=True,
        hicache_storage_backend="kvcr",
    )
    assert metadata[KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY] == {
        "0": "tcp://127.0.0.1:32100",
        "1": "tcp://127.0.0.1:32102",
    }
    _, metadata = _linker_registration(extra)
    assert not metadata


@pytest.mark.parametrize(
    "filename, content, error",
    [
        (None, "{", ValueError),
        (None, "[]", ValueError),
        ("extra.ini", "control_port = 29000", ValueError),
        ("missing.json", None, OSError),
    ],
)
def test_registration_rejects_invalid_config(tmp_path, filename, content, error):
    extra = content
    if filename is not None:
        path = tmp_path / filename
        if content is not None:
            path.write_text(content, encoding="utf-8")
        extra = f"@{path}"
    with pytest.raises(error):
        _linker_registration(extra)


@pytest.mark.parametrize(
    "output_type, worker_type, forwards_kv_hints",
    [
        (ModelType.Chat | ModelType.Completions, WorkerType.Aggregated, False),
        (ModelType.Chat | ModelType.Completions, WorkerType.Decode, True),
        (ModelType.Prefill, WorkerType.Prefill, True),
    ],
)
def test_registration_requires_handler_hint_forwarding(
    monkeypatch, output_type, worker_type, forwards_kv_hints
):
    args, metadata = _linker_registration(
        {"advertise_host": "192.0.2.1", "control_port": 29000}
    )
    args.model_path = args.served_model_name = "test-model"
    args.page_size = 64
    metadata.clear()
    runtime = SimpleNamespace(
        set_engine_specific=lambda key, value: metadata.__setitem__(
            key, json.loads(value)
        )
    )
    monkeypatch.setattr(register, "get_runtime_config", AsyncMock(return_value=runtime))
    register_model = AsyncMock()
    monkeypatch.setattr(register, "register_model", register_model)
    dynamo_args = SimpleNamespace(
        use_sglang_tokenizer=False, model_source_uri=None, custom_jinja_template=None
    )
    kwargs = {"forwards_kv_hints": True} if forwards_kv_hints else {}
    asyncio.run(
        register.register_model_with_readiness_gate(
            _HintEngine(),
            None,
            args,
            dynamo_args,
            output_type=output_type,
            worker_type=worker_type,
            **kwargs,
        )
    )
    register_model.assert_awaited_once()
    assert (KV_HINT_TRANSFER_CAPABILITY_KEY in metadata) is forwards_kv_hints
    assert (
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY in metadata
    ) is forwards_kv_hints
