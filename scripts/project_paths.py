"""專案路徑常數（以 repo 根目錄為基準）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "paper_trading"
