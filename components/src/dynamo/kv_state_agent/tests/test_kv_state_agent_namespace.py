# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

pytest.importorskip("dynamo.llm")

from dynamo.common.utils.namespace import get_worker_namespace  # noqa: E402
from dynamo.kv_state_agent import __main__ as host_main  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _Stop(Exception):
    pass


async def _resolved_namespace(monkeypatch) -> str:
    """Run the host entry point until it builds its control endpoint."""
    paths = []

    class FakeRuntime:
        def endpoint(self, path):
            paths.append(path)
            return path

    def fake_host(endpoint, max_slots, slots_per_socket):
        raise _Stop

    monkeypatch.setattr(host_main, "KvStateAgentHost", fake_host)
    monkeypatch.setattr(sys, "argv", ["kv_state_agent", "--max-slots", "2"])
    with pytest.raises(_Stop):
        await host_main.worker.__wrapped__(FakeRuntime())
    assert len(paths) == 1 and paths[0].endswith(".kv_state_agent.control")
    return paths[0].removesuffix(".kv_state_agent.control")


@pytest.mark.asyncio
async def test_host_registers_under_the_worker_suffixed_namespace(monkeypatch):
    # The operator sets this suffix on worker components (KvStateAgent included);
    # vLLM workers discover the host in the suffixed namespace.
    monkeypatch.setenv("DYN_NAMESPACE", "x")
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "abc")

    namespace = await _resolved_namespace(monkeypatch)

    assert namespace == "x-abc" == get_worker_namespace()


@pytest.mark.asyncio
async def test_host_namespace_is_unchanged_without_a_suffix(monkeypatch):
    monkeypatch.setenv("DYN_NAMESPACE", "x")
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)

    assert await _resolved_namespace(monkeypatch) == "x"
