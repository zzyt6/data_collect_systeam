"""Piper 子项目路径辅助工具。

Piper 数采代码位于仓库的 piper/ 子目录中；本地 piper_sdk 也作为子模块放在这里。
这里负责在直接运行脚本时把 SDK 加入导入路径，同时给采集数据提供仓库级 data/ 目录。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
LOCAL_SDK_ROOT = REPO_ROOT / "piper_sdk"


def ensure_local_piper_sdk() -> None:
    """如果本地存在 piper_sdk 源码，就把它加入 sys.path 前面。"""

    if (LOCAL_SDK_ROOT / "piper_sdk" / "__init__.py").exists():
        sdk_path = str(LOCAL_SDK_ROOT)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
