# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared loader for SGLang HiCache storage configuration files."""

import json
import os
import sys
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
