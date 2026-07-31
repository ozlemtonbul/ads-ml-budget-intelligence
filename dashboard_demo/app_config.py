from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Public Streamlit demo uygulaması yalnızca anonimleştirilmiş
# demo verilerinin bulunduğu klasörü kullanır.
OUTPUT_DIR = (PROJECT_ROOT / "demo_data").resolve()

APP_TITLE = "Ads Budget Intelligence AI Agent"
APP_SUBTITLE = "AI-Powered Advertising Decision Intelligence Platform"

CURRENCY_SYMBOL = "₺"
DEFAULT_TARGET_ROAS = 3.0