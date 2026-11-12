<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Guard as a Service

**Experimental.** These templates wire a NVIDIA Dynamo worker to a separately
managed KVCR holder. They are preparation inputs, not a validated GPU serving
recipe. Do not deploy the unsubstituted templates.

The node broker creates a holder Pod for a named pool. The holder owns and
pre-touches its private memory-backed pool and runs the existing KVCR service.
The init container requests that pool and writes its descriptor to an `emptyDir`.
The worker connects directly to the holder's socket using KVCR's existing FD
claim and recovery protocol. Its control listener advertises the holder's Pod IP,
not the worker's IP. No host networking or shared PID namespace is used.

## Preparation Inputs

The DaemonSet runs the v2 supervisor (`kvcr-service`, `kvcr.service.server`) on
`${GAAS_SOCKET_ROOT}/service.sock`, with hostPID so it can read each claimant's
pod UID. The worker's `init` waits for that socket and `launch` claims through
it with `--tenant` and `--lineage` (the DGD name); the supervisor creates the
holder inside the claim. The worker also passes `--state-agent-namespace ${GAAS_NAMESPACE}-${GAAS_NAME}`,
the operator's Dynamo namespace for the graph (`ComputeDynamoNamespace`, valid
while the DGD name has no dots, which the operator turns into dashes, and the
result fits a 63-character label), so the supervisor enforces the
`nvidia.com/dynamo-namespace` label as well as the lineage label. Without those flags the worker uses the v1 broker path.

`guard-daemonset.yaml` contains a namespaced ServiceAccount, Role, RoleBinding
and a node-pinned DaemonSet. `dgd.yaml` adds the worker init gate through
`extraPodSpec`. `manifest-bundle.json` contains the same five resources for a
stdlib-only preparation tool; the template-consistency test prevents drift.

Render all of these variables before reviewing a deployment:

| Variable | Meaning |
| --- | --- |
| `GAAS_NAME` | DNS-label deployment and manager identity |
| `GAAS_NAMESPACE` | Existing namespace approved for this experiment |
| `GAAS_NODE` | One approved node hostname |
| `GAAS_SOCKET_ROOT` | Short private directory under `/node-shm`; all socket paths must fit Linux's 108-byte limit |
| `GAAS_IMAGE` | Consumer image reference pinned by digest, including the matching KVCR and `dynamo.gaas` changes |
| `GAAS_COMPATIBILITY_DIGEST` | Stable model, layout and cache-format compatibility identity |
| `GAAS_MODEL`, `GAAS_MODEL_REVISION` | Reviewed model id (served name); weights load from the `shared-model-cache` PVC at `/model-cache/Qwen3.8-27B-FP8`; the revision is no longer used |
| `GAAS_GPU_KV_BYTES`, `GAAS_MAX_MODEL_LEN` | Reviewed GPU KV-cache capacity and context limit |
| `GAAS_POOL_SIZES_GB` | Supervisor `--pool-sizes-gb` (the pool layout; the supervisor runs three Guards, `--guard-count 3`) |
| `GAAS_SLOT_CAP_BYTES` | Supervisor `--slot-cap-bytes`, the node's slot capacity |
| `GAAS_CPU_PRIMARY_BYTES` | CPU primary-tier capacity; distinct from the holder's secondary-pool capacity; vLLM places this region in the worker's `/dev/shm`, so the template sets `sharedMemory.size: 64Gi` (the operator sizes `/dev/shm` from it) and a 96Gi worker memory limit (raise both with larger values) |
| `GAAS_SUPERVISOR_MEMORY` | `kvcr-service` container memory request and limit (required, no default; the render raises KeyError if unset). The supervisor commits one in-process pool per Guard rank at startup, charged to this pod, so size it at guard-count x (pool size + 100 MiB) plus ~0.5 GiB: e.g. `4Gi` for guard-count 3 with 1 GiB pools, ~`104Gi` for 50 GiB pools |
| `GAAS_CLAIM_GB` | v2 claim size (required, no default; the render raises KeyError if unset), used for `--size-gb` (total), `--minimum-gb` and `--maximum-gb` (exact grant) of every `gaas-claim` init container and worker. GiB (binary, same unit as `--pool-sizes-gb`); on the S4b path set it to `GAAS_CPU_PRIMARY_BYTES` / 2^30 so the grant equals the engine's CPU pool exactly; partial grants are refused |

The sample claims pool `h7-pool-a` with `GAAS_CLAIM_GB` GiB of data. A separate smaller claim
(0.5 GiB minimum) can exercise independent reclamation. Each holder adds KVCR's 100 MiB journal
and runtime headroom to its memory reservation. The service has three Guards (guard-count 3, slots 0 to 2, ports 55001 to 55003) and
each of the three workers (`VllmWorkerA`, `VllmWorkerB`, `VllmWorkerC`) uses TP=PP=DP=1 and its own Guard: `--guard-index N` and control port `55001 + N` (override with `--control-port`). The DaemonSet runs on the host network and the workers advertise the node's address through `GAAS_ADVERTISE_HOST` (`--advertise-host`); a v2 launch exits without it. `--state-agent` attaches each worker to the `KvStateAgent` service.
Runtime block layout comes from the worker's KV view stride, not the pool's
capacity.

## Claim and Launch

The init gate calls `ensure_pool` with `pool_id`, `size_gb` and
`compatibility_digest`. It accepts only a ready descriptor with matching
identity, holder UID, holder IP and a service socket inside the broker directory.
It atomically writes a mode-0600 descriptor. The main container mounts that
descriptor read-only.

The launcher requires the same identity and refuses conflicting existing
offload settings. It passes the pinned vLLM tier schema through
`--kv-transfer-config`: `OffloadingConnector`, `TieringOffloadingSpec` and one
KVCR secondary tier. Existing state-agent and telemetry configuration is
preserved. Complete routing, KV-event and workload configuration must still be
validated for the intended serving test; no production routing behavior is
established by these templates.

Protected options accept complete hyphenated or underscored names and the
`-tp`, `-pp`, and `-dp` aliases. The launcher rejects their abbreviations,
dotted transfer-config updates, custom connector modules, external `--config`
files, and another backend `--` separator. Supply transfer settings as one
whole JSON object so native parser preprocessing cannot override the claim.

## Release and Failure Handling

Holders are label-managed and have no worker or DaemonSet owner reference.
Worker exit leaves the pool reserved for reattachment. Explicit pool release
asks the managing Guard to drain. The holder reservation remains until pending
grants, primary leases and every external mapping are gone. Busy, failed or
uncertain drain results retain it. Only terminal drain acknowledgement permits
the broker's UID-preconditioned Pod deletion. Broker restart reconciles the
persisted pool identity and release intent; it does not delete by timeout.

This does not protect against administrator-forced deletion or node eviction.
Teardown must drain workers and pools before removing the broker and its RBAC.
Do not attach a garbage-collecting owner reference to holders.

## Policy and Validation Scope

The shared **socket directory** uses a `/dev/shm` hostPath. The worker never
mounts the holder's backing file. PSA Baseline forbids Pod-spec hostPath volumes;
this template is therefore not a Baseline-compatible delivery solution. A
platform-managed device plugin or DRA/CDI directory mount requires separate
implementation and runtime validation.

CPU roles disable GPU visibility. The worker template requests three single-GPU workers only;
deploy it through a reviewed namespace-fenced launcher after resource approval.
No cluster or GPU run is required to execute the wrapper's unit tests.

Build the consumer from the pinned Dynamo vLLM nightly and overlay only the new
`dynamo/gaas` package plus the matching KVCR wheel. Preserve and validate the
installed Dynamo native module and vLLM adapter hashes. The wrapper does not
replace vLLM or rebuild native Dynamo.
