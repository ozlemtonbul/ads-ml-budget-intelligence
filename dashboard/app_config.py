from __future__ import annotations

from pathlib import Path

from config.settings import (
    OUTPUT_DIR as SETTINGS_OUTPUT_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CONFIGURED_OUTPUT_DIR = Path(
    SETTINGS_OUTPUT_DIR
).expanduser()

OUTPUT_DIR = (
    _CONFIGURED_OUTPUT_DIR
    if _CONFIGURED_OUTPUT_DIR.is_absolute()
    else PROJECT_ROOT / _CONFIGURED_OUTPUT_DIR
).resolve()

APP_TITLE = "Ads Budget Intelligence AI Agent"
APP_SUBTITLE = "AI-Powered Advertising Decision Intelligence Platform"

CURRENCY_SYMBOL = "₺"
DEFAULT_TARGET_ROAS = 3.0
