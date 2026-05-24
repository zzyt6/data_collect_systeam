#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Piper XY 遥控、关节角 action 数据采集器的轻量启动入口。
使用方法：python piper/tools/diffusion_xy_collector.py \
--can can0 \
--camera-wrist 10 \
--camera-global 4 \
--initial-pose-json piper/initial_pose/initial_pose.json \
--connect \
--hz 10 \
--command-hz 30 \
--xy-speed-mm-s 50 \
--speed-percent 50 \
--reset-duration 5 \
--reset-hz 20 \
--disable-delay 10 \
--output-dir data/piper_xy

"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piper_diffusion_collect.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
