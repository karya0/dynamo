// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests.

use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwnerKey, ResidencyRoutingSnapshot, WorkerWithDpRank,
};

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
// TODO: Rename these constants and wire values with the matching KVCC names.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime-data keys used to build transfer hints.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

const KV_HINT_PROTOCOL_VERSION: &str = "0.1";
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";
const KV_FETCH_ACTION_VERSION: &str = "1.0";

/// Typed payload for the `kv.fetch@1.0` point-to-point action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvSourceLocationsPayload {
    pub source_control_endpoint: String,
    /// Source-side KV block hashes available for this request. These may describe
    /// a suffix when the target already has the preceding prefix; consumers use
    /// hash membership rather than interpreting vector positions as token offsets.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

/// One versioned action in a [`KvHint`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHintAction {
    pub action_id: String,
    pub action_type: String,
    pub action_version: String,
    pub payload: BTreeMap<String, serde_json::Value>,
}

impl KvHintAction {
    pub fn new(
        action_id: impl Into<String>,
        action_type: impl Into<String>,
        action_version: impl Into<String>,
        payload: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            action_id: action_id.into(),
            action_type: action_type.into(),
            action_version: action_version.into(),
            payload,
        }
    }

    pub fn fetch(action_id: impl Into<String>, payload: KvSourceLocationsPayload) -> Self {
        let KvSourceLocationsPayload {
            source_control_endpoint,
            block_hashes,
        } = payload;
        Self::new(
            action_id,
            KV_FETCH_ACTION_TYPE,
            KV_FETCH_ACTION_VERSION,
            BTreeMap::from([
                (
                    "source_control_endpoint".to_string(),
                    serde_json::json!(source_control_endpoint),
                ),
                ("block_hashes".to_string(), serde_json::json!(block_hashes)),
            ]),
        )
    }
}

/// One versioned KV hint message for the selected backend request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHint {
    pub protocol_version: String,
    pub message_id: String,
    pub actions: Vec<KvHintAction>,
}

impl KvHint {
    pub fn new(message_id: impl Into<String>, actions: Vec<KvHintAction>) -> Self {
        Self {
            protocol_version: KV_HINT_PROTOCOL_VERSION.to_string(),
            message_id: message_id.into(),
            actions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KvTransferCandidateSource {
    Worker(WorkerWithDpRank),
    CacheOwner(ResidencyOwnerKey),
}

impl From<WorkerWithDpRank> for KvTransferCandidateSource {
    fn from(worker: WorkerWithDpRank) -> Self {
        Self::Worker(worker)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvTransferRange {
    pub source: KvTransferCandidateSource,
    pub start_block: usize,
    pub parent_hash: ExternalSequenceBlockHash,
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

impl KvTransferRange {
    pub(crate) fn matches_chain(&self, chain: &[ExternalSequenceBlockHash]) -> bool {
        let Some(parent_pos) = self.start_block.checked_sub(1) else {
            return false;
        };
        !self.block_hashes.is_empty()
            && self
                .start_block
                .checked_add(self.block_hashes.len())
                .is_some()
            && chain.get(parent_pos) == Some(&self.parent_hash)
            && self
                .block_hashes
                .iter()
                .zip(chain.iter().skip(self.start_block))
                .all(|(source, canonical)| source == canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvTransferCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(KvTransferCandidateSource, usize)>,
    /// Contiguous source-owned suffixes, separate from root-aligned prefixes.
    pub owner_ranges: Vec<KvTransferRange>,
    pub routing_snapshot: Option<Arc<ResidencyRoutingSnapshot>>,
}

impl KvTransferCandidates {
    pub fn best_source<F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut is_eligible_source: F,
    ) -> Option<(KvTransferCandidateSource, Vec<ExternalSequenceBlockHash>)>
    where
        F: FnMut(KvTransferCandidateSource) -> bool,
    {
        let prefix = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(source, blocks)| {
                *blocks > prefix_blocks_to_beat && is_eligible_source(*source)
            })
            .max_by(|(left_source, left_blocks), (right_source, right_blocks)| {
                left_blocks
                    .cmp(right_blocks)
                    .then_with(|| right_source.cmp(left_source))
            });
        let range = self
            .owner_ranges
            .iter()
            .filter_map(|range| {
                let end = range.start_block.checked_add(range.block_hashes.len())?;
                (range.matches_chain(&self.block_hashes)
                    && range.start_block <= prefix_blocks_to_beat
                    && prefix_blocks_to_beat < end
                    && is_eligible_source(range.source))
                .then_some((range, end))
            })
            .max_by(|(left, left_end), (right, right_end)| {
                left_end
                    .cmp(right_end)
                    .then_with(|| right.source.cmp(&left.source))
                    .then_with(|| left.start_block.cmp(&right.start_block))
            });

        if let Some((range, end)) = range {
            // Preserve legacy prefix payloads on equal-length matches.
            if prefix.is_none_or(|(_, prefix_end)| end > prefix_end) {
                return Some((range.source, range.block_hashes.clone()));
            }
        }
        let (source, prefix_blocks) = prefix?;
        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suffix_candidates() -> KvTransferCandidates {
        KvTransferCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ],
            owner_prefix_blocks: Vec::new(),
            owner_ranges: vec![KvTransferRange {
                source: WorkerWithDpRank::new(7, 0).into(),
                start_block: 2,
                parent_hash: ExternalSequenceBlockHash(102),
                block_hashes: vec![
                    ExternalSequenceBlockHash(103),
                    ExternalSequenceBlockHash(104),
                ],
            }],
            routing_snapshot: None,
        }
    }

    #[test]
    fn suffix_range_requires_receiver_to_cover_gap() {
        let candidates = suffix_candidates();
        assert!(candidates.best_source(0, |_| true).is_none());
        assert!(candidates.best_source(1, |_| true).is_none());
        for prefix in [2, 3] {
            assert_eq!(
                candidates.best_source(prefix, |_| true),
                Some((
                    WorkerWithDpRank::new(7, 0).into(),
                    vec![
                        ExternalSequenceBlockHash(103),
                        ExternalSequenceBlockHash(104)
                    ],
                ))
            );
        }
        assert!(candidates.best_source(4, |_| true).is_none());
        assert!(candidates.best_source(2, |_| false).is_none());
    }

    #[test]
    fn suffix_range_rejects_wrong_parent_and_conflicting_chain() {
        let mut candidates = suffix_candidates();
        candidates.owner_ranges[0].parent_hash = ExternalSequenceBlockHash(999);
        assert!(candidates.best_source(2, |_| true).is_none());
        candidates.owner_ranges[0].parent_hash = ExternalSequenceBlockHash(102);
        candidates.block_hashes.push(ExternalSequenceBlockHash(999));
        assert!(candidates.best_source(2, |_| true).is_none());
    }

    #[test]
    fn suffix_ranges_do_not_bridge_an_uncovered_hole() {
        let mut candidates = suffix_candidates();
        candidates.block_hashes = (101..=105).map(ExternalSequenceBlockHash).collect();
        candidates.owner_ranges.push(KvTransferRange {
            source: WorkerWithDpRank::new(7, 0).into(),
            start_block: 5,
            parent_hash: ExternalSequenceBlockHash(105),
            block_hashes: vec![
                ExternalSequenceBlockHash(106),
                ExternalSequenceBlockHash(107),
            ],
        });
        assert!(candidates.best_source(4, |_| true).is_none());
        assert_eq!(
            candidates.best_source(5, |_| true),
            Some((
                WorkerWithDpRank::new(7, 0).into(),
                vec![
                    ExternalSequenceBlockHash(106),
                    ExternalSequenceBlockHash(107)
                ],
            ))
        );
    }

    #[test]
    fn suffix_range_preserves_legacy_prefix_ties_and_fallback() {
        let mut candidates = suffix_candidates();
        candidates.block_hashes.extend([
            ExternalSequenceBlockHash(103),
            ExternalSequenceBlockHash(104),
        ]);
        let legacy = WorkerWithDpRank::new(9, 0).into();
        candidates.owner_prefix_blocks.push((legacy, 4));
        assert_eq!(
            candidates.best_source(2, |_| true),
            Some((legacy, candidates.block_hashes.clone()))
        );
        candidates.owner_ranges[0].parent_hash = ExternalSequenceBlockHash(999);
        assert_eq!(
            candidates.best_source(2, |_| true),
            Some((legacy, candidates.block_hashes.clone()))
        );
    }

    #[test]
    fn suffix_range_rejects_empty_missing_prefix_and_overflow() {
        let mut candidates = suffix_candidates();
        candidates.owner_ranges[0].block_hashes.clear();
        assert!(candidates.best_source(2, |_| true).is_none());
        let mut candidates = suffix_candidates();
        candidates.block_hashes.truncate(1);
        assert!(candidates.best_source(2, |_| true).is_none());
        let mut candidates = suffix_candidates();
        candidates.owner_ranges[0].start_block = usize::MAX;
        assert!(candidates.best_source(usize::MAX, |_| true).is_none());
    }

    #[test]
    fn serializes_versioned_fetch_action() {
        let hint = KvHint::new(
            "msg-123",
            vec![KvHintAction::fetch(
                "a1",
                KvSourceLocationsPayload {
                    source_control_endpoint: "tcp://127.0.0.1:23280".to_string(),
                    block_hashes: vec![
                        ExternalSequenceBlockHash(11),
                        ExternalSequenceBlockHash(22),
                    ],
                },
            )],
        );

        assert_eq!(
            serde_json::to_value(hint).unwrap(),
            serde_json::json!({
                "protocol_version": "0.1",
                "message_id": "msg-123",
                "actions": [{
                    "action_id": "a1",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://127.0.0.1:23280",
                        "block_hashes": [11, 22],
                    },
                }],
            })
        );
    }

    #[test]
    fn deserializes_unknown_action_as_opaque_transport_data() {
        let value = serde_json::json!({
            "protocol_version": "0.1",
            "message_id": "msg-future",
            "actions": [{
                "action_id": "a-future",
                "action_type": "kv.future_action",
                "action_version": "7.3",
                "payload": {
                    "nested": {"enabled": true},
                    "items": [1, 2, 3],
                },
            }],
        });

        let hint: KvHint = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(hint.actions[0].action_type, "kv.future_action");
        assert_eq!(serde_json::to_value(hint).unwrap(), value);
    }

    #[test]
    fn best_source_selects_longest_eligible_prefix() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let excluded = WorkerWithDpRank::new(9, 0);
        let candidates = KvTransferCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ],
            owner_prefix_blocks: vec![
                (worker_b.into(), 2),
                (excluded.into(), 3),
                (worker_a.into(), 3),
            ],
            owner_ranges: Vec::new(),
            routing_snapshot: None,
        };

        let selected = candidates.best_source(0, |source| source != excluded.into());

        assert_eq!(
            selected,
            Some((
                worker_a.into(),
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                ],
            ))
        );
    }

    #[test]
    fn best_source_fails_closed_on_invalid_prefix_length() {
        let candidates = KvTransferCandidates {
            block_hashes: vec![ExternalSequenceBlockHash(101)],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 0).into(), 2)],
            owner_ranges: Vec::new(),
            routing_snapshot: None,
        };

        assert!(candidates.best_source(0, |_| true).is_none());
    }

    #[test]
    fn best_source_requires_prefix_longer_than_threshold() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let candidates = KvTransferCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
                ExternalSequenceBlockHash(104),
            ],
            owner_prefix_blocks: vec![(worker_a.into(), 3), (worker_b.into(), 4)],
            owner_ranges: Vec::new(),
            routing_snapshot: None,
        };

        assert!(
            candidates
                .best_source(3, |source| source == worker_a.into())
                .is_none()
        );
        assert_eq!(
            candidates.best_source(3, |_| true),
            Some((
                worker_b.into(),
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                    ExternalSequenceBlockHash(104),
                ],
            ))
        );
    }
}
