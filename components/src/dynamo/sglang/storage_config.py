# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared parsing for SGLang HiCache storage configuration."""

import json
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_hicache_storage_extra_config_file(path: str) -> Any:
    """Load an ``@path`` config in SGLang's supported JSON/TOML/YAML formats."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if ext == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)
    if ext in (".yaml", ".yml"):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError(f"unsupported config file {path!r} (format {ext!r})")


def parse_hicache_storage_extra_config(
    raw_extra_config: Any, *, strict: bool = False
) -> dict[str, Any]:
    """Parse inline or ``@file`` config; strict callers retain configuration errors."""
    try:
        if isinstance(raw_extra_config, str):
            raw_extra_config = raw_extra_config.strip()
            if not raw_extra_config:
                return {}
            raw_extra_config = (
                load_hicache_storage_extra_config_file(raw_extra_config[1:])
                if raw_extra_config.startswith("@")
                else json.loads(raw_extra_config)
            )
        if raw_extra_config is None:
            return {}
        if not isinstance(raw_extra_config, Mapping):
            raise ValueError("hicache_storage_backend_extra_config must be an object")
        return dict(raw_extra_config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        if strict:
            raise
        logging.warning("Failed to parse hicache_storage_backend_extra_config: %s", exc)
        return {}
