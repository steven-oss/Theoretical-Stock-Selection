"""Jupyter notebook 路徑初始化。"""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap():
    """設定 sys.path、載入 .env，回傳 (ROOT, DATA_DIR, PAPER_DIR)。"""
    cwd = Path.cwd()
    root = cwd if (cwd / "scripts" / "project_paths.py").exists() else cwd.parent
    if not (root / "scripts" / "project_paths.py").exists():
        raise FileNotFoundError(
            "找不到專案根目錄。請在 pythonFinmind/ 或 pythonFinmind/notebooks/ 下執行 notebook。"
            f" 目前 cwd={cwd}"
        )

    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    from dotenv import load_dotenv
    from project_paths import DATA_DIR, PAPER_DIR, ROOT

    load_dotenv(ROOT / ".env")
    DATA_DIR.mkdir(exist_ok=True)
    return ROOT, DATA_DIR, PAPER_DIR
