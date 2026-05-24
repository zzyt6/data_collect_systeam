"""项目路径辅助工具。

仓库里包含本地 piper_sdk 源码。这里负责在直接运行脚本时把它加入导入路径，
这样不需要先做 editable pip install。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SDK_ROOT = REPO_ROOT / "piper_sdk"


def ensure_local_piper_sdk() -> None:
    """如果本地存在 piper_sdk 源码，就把它加入 sys.path 前面。"""

    if (LOCAL_SDK_ROOT / "piper_sdk" / "__init__.py").exists():
        sdk_path = str(LOCAL_SDK_ROOT)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
