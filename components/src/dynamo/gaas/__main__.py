# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvcr.service.node_service import ensure_pool

from dynamo.gaas import main

if __name__ == "__main__":
    raise SystemExit(main(ensure_pool=ensure_pool))
