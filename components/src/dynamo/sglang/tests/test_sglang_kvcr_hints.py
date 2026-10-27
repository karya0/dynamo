# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router hint forwarding, HiCache endpoint layout, and registration gating."""

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
from dynamo.sglang.kv_hints import kv_hint_kwargs, kvcr_hint_endpoints

# Marker-only collection may provide only partial SGLang stubs.
_register_import_error = None
try:
    from dynamo.sglang import register
except ImportError as exc:
    register = None
    _register_import_error = exc

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
    assert kv_hint_kwargs(_HintEngine(), {"nvext": {"kv_hint": hint}}) == {}


@pytest.mark.parametrize(
    "host, control_port, expected_host",
    [
        ("192.0.2.1", 28000, "192.0.2.1"),
        ("2001:db8::1", 28000, "[2001:db8::1]"),
        (None, 28000, None),
        ("0.0.0.0", 28000, None),
        ("tcp://192.0.2.1", 28000, None),
        ("192.0.2.1", True, None),
        ("192.0.2.1", 65535, None),
    ],
)
def test_hicache_endpoint_layout(host, control_port, expected_host):
    # Pure address arithmetic: these tests do not bind sockets.
    kwargs = dict(
        extra_config={"control_advertise_host": host, "control_port": control_port},
        dp_size=2,
        tp_size=2,
        attn_cp_size=2,
        pp_size=1,
    )
    if expected_host is None:
        with pytest.raises(ValueError):
            kvcr_hint_endpoints(**kwargs)
    else:
        assert kvcr_hint_endpoints(**kwargs) == {
            "0": f"tcp://{expected_host}:{control_port}",
            "1": f"tcp://{expected_host}:{control_port + 4}",
        }


@pytest.mark.parametrize(
    "change, forwards_kv_hints, engine_type",
    [
        ({}, False, _HintEngine),
        ({}, True, _HintEngine),
        ({}, True, _OldEngine),
        ({"nnodes": 2}, True, _HintEngine),
        ({"pp_size": 2}, True, _HintEngine),
        ({"hicache_storage_backend": "mooncake"}, True, _HintEngine),
        (
            {
                "enable_hierarchical_cache": False,
                "enable_unified_cache_external_linker": True,
                "unified_cache_external_linker_backend": "kvcr",
            },
            True,
            _HintEngine,
        ),
        ({"hicache_storage_backend_extra_config": " \n\t "}, True, _HintEngine),
        ({"hicache_storage_backend_extra_config": {}}, True, _HintEngine),
        (
            {
                "hicache_storage_backend_extra_config": {
                    "control_advertise_host": "192.0.2.1",
                    "enable_remote_hint": False,
                }
            },
            True,
            _HintEngine,
        ),
    ],
)
def test_registration_requires_a_hint_forwarding_hicache_worker(
    monkeypatch, tmp_path, change, forwards_kv_hints, engine_type
):
    if _register_import_error is not None:
        raise _register_import_error
    control_port = 28000
    path = tmp_path / "extra.json"
    path.write_text(
        json.dumps(
            {
                "control_advertise_host": "192.0.2.1",
                "control_port": control_port,
                "enable_remote_hint": True,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        **{
            "enable_hierarchical_cache": True,
            "hicache_storage_backend": "kvcr",
            "nnodes": 1,
            "attn_dp_size": 2,
            "attn_cp_size": 1,
            "tp_size": 4,
            "pp_size": 1,
            "page_size": 64,
            "model_path": "test-model",
            "served_model_name": "test-model",
            "hicache_storage_backend_extra_config": f"  @{path}  ",
            **change,
        }
    )
    metadata = {}
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
            engine_type(),
            None,
            args,
            dynamo_args,
            output_type=ModelType.Chat | ModelType.Completions,
            worker_type=WorkerType.Aggregated,
            **kwargs,
        )
    )
    register_model.assert_awaited_once()
    metadata = {
        key: value
        for key, value in metadata.items()
        if key
        in (
            KV_HINT_TRANSFER_CAPABILITY_KEY,
            KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        )
    }
    if not change and forwards_kv_hints and engine_type is _HintEngine:
        assert metadata == {
            KV_HINT_TRANSFER_CAPABILITY_KEY: True,
            KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: "aggregated",
            KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: {
                "0": f"tcp://192.0.2.1:{control_port}",
                "1": f"tcp://192.0.2.1:{control_port + 2}",
            },
        }
    else:
        assert not metadata
