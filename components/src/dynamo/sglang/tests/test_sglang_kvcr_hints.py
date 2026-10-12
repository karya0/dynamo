# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router-selected KV hints reach the in-process SGLang engine unchanged."""

import json
from types import SimpleNamespace

import pytest

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
)
from dynamo.llm import WorkerType
from dynamo.sglang.kv_hints import (
    kv_hint_kwargs,
    kvcr_hint_endpoints,
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
    # Do not turn arbitrary client extra_args into a trusted router decision.
    assert kv_hint_kwargs(_HintEngine(), {"extra_args": {"kv_hint": hint}}) == {}


def test_endpoint_layout_matches_attention_group_offsets():
    # No sockets are opened: this pins the arithmetic/protocol, not allocation.
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
