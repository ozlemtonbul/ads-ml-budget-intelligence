from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)


from dashboard_demo.app_config import APP_TITLE, OUTPUT_DIR
from dashboard_demo.layout import (
    initialize_dashboard,
    localized_text,
    render_read_only_footer,
)
from dashboard_demo.services.executive_data import (
    get_available_date_bounds,
    load_executive_data,
)
from dashboard_demo.utils import get_latest_output_time
from src.llm.manager import get_llm_runtime_info


def count_output_files() -> int:
    """Return the number of generated output files."""

    if not OUTPUT_DIR.exists():
        return 0

    return sum(
        1
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
    )


def render_navigation_card(
    title: str,
    description: str,
    page_path: str | None,
    button_label: str,
    key: str,
) -> None:
    """Render one dashboard navigation card."""

    with st.container(border=True):
        st.subheader(title)
        st.caption(description)

        if page_path is None:
            st.button(
                button_label,
                disabled=True,
                width="stretch",
                key=key,
            )
            return

        if st.button(
            button_label,
            width="stretch",
            key=key,
        ):
            st.switch_page(page_path)


# Load only the data required to establish the dashboard reference date.
data = load_executive_data()
available_start, available_end = get_available_date_bounds(
    data.daily
)

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=APP_TITLE,
    page_icon="◈",
    title=APP_TITLE,
    subtitle=(
        "Google Ads ve GA4 çıktılarını tek bir karar destek "
        "katmanında birleştiren reklam bütçe zekâsı platformu."
        if initial_language == "tr"
        else
        "An advertising budget intelligence platform that "
        "combines Google Ads and GA4 outputs in one decision "
        "support layer."
    ),
    eyebrow="AI Campaign Intelligence Platform",
    default_preset="last_30_days",
    default_comparison="previous_period",
    reference_date=available_end,
)

language = context.language
runtime_info = get_llm_runtime_info()


st.subheader(
    localized_text(
        language,
        "Platform Durumu",
        "Platform Status",
    )
)

daily_row_count = len(data.daily)
daily_day_count = 0

if not data.daily.empty and "Date" in data.daily.columns:
    daily_day_count = int(
        pd.to_datetime(
            data.daily["Date"],
            errors="coerce",
        )
        .dropna()
        .dt.date
        .nunique()
    )

first_status_row = st.columns(3)

first_status_row[0].metric(
    localized_text(
        language,
        "Günlük Veri Satırı",
        "Daily Data Rows",
    ),
    f"{daily_row_count:,}",
)

first_status_row[1].metric(
    localized_text(
        language,
        "Veri Bulunan Gün",
        "Days With Data",
    ),
    daily_day_count,
)

first_status_row[2].metric(
    localized_text(
        language,
        "Üretilen Çıktı",
        "Generated Outputs",
    ),
    count_output_files(),
)

second_status_row = st.columns(3)

date_period = (
    f"{available_start:%d.%m.%Y} – "
    f"{available_end:%d.%m.%Y}"
    if available_start is not None
    and available_end is not None
    else localized_text(
        language,
        "Veri bulunamadı",
        "No data found",
    )
)

second_status_row[0].metric(
    localized_text(
        language,
        "Mevcut Veri Dönemi",
        "Available Data Period",
    ),
    date_period,
)

llm_is_ready = bool(
    runtime_info.get("ready", False)
)

second_status_row[1].metric(
    localized_text(
        language,
        "AI Çalışma Modu",
        "AI Runtime Mode",
    ),
    localized_text(
        language,
        "Hibrit LLM"
        if llm_is_ready
        else "Deterministik",
        "Hybrid LLM"
        if llm_is_ready
        else "Deterministic",
    ),
)

second_status_row[2].metric(
    localized_text(
        language,
        "Son Çıktı",
        "Latest Output",
    ),
    get_latest_output_time(),
)

if not llm_is_ready:
    st.info(
        localized_text(
            language,
            (
                "API anahtarı bulunmadığı için sistem deterministik "
                "analiz modunda çalışıyor. KPI, grafik, model ve "
                "optimizasyon hesapları kullanılabilir."
            ),
            (
                "The system is running in deterministic analysis "
                "mode because no API key is configured. KPI, chart, "
                "model and optimization calculations remain available."
            ),
        ),
        icon="ℹ️",
    )

st.divider()

st.subheader(
    localized_text(
        language,
        "Analiz Alanları",
        "Analysis Areas",
    )
)

first_navigation_row = st.columns(2)

with first_navigation_row[0]:
    render_navigation_card(
        title=localized_text(
            language,
            "Yönetici Özeti",
            "Executive Overview",
        ),
        description=localized_text(
            language,
            (
                "Gerçekleşen KPI'ları, veri kapsamını, hedef ROAS'ı, "
                "riskleri ve bütçe aksiyonlarını inceleyin."
            ),
            (
                "Review actual KPIs, data coverage, target ROAS, "
                "risks and budget actions."
            ),
        ),
        page_path="pages/1_Executive_Overview.py",
        button_label=localized_text(
            language,
            "Yönetici Özetini Aç",
            "Open Executive Overview",
        ),
        key="home_open_executive",
    )

with first_navigation_row[1]:
    render_navigation_card(
        title=localized_text(
            language,
            "AI Asistanı",
            "AI Assistant",
        ),
        description=localized_text(
            language,
            (
                "Reklam verileri ve optimizasyon çıktıları hakkında "
                "doğal dilde soru sorun."
            ),
            (
                "Ask natural-language questions about advertising "
                "data and optimization outputs."
            ),
        ),
        page_path="pages/5_Ask_AI.py",
        button_label=localized_text(
            language,
            "AI Asistanını Aç",
            "Open AI Assistant",
        ),
        key="home_open_ai",
    )

second_navigation_row = st.columns(3)

second_navigation_row = st.columns(3)

navigation_pages = [
    {
        "title": localized_text(
            language,
            "Kampanya Analizi",
            "Campaign Analysis",
        ),
        "description": localized_text(
            language,
            (
                "Kampanya bazında performans, ROAS, harcama, "
                "gelir ve dönüşüm analizlerini inceleyin."
            ),
            (
                "Review campaign-level performance, ROAS, spend, "
                "revenue and conversion analysis."
            ),
        ),
        "page_path": "pages/2_Campaign_Analysis.py",
        "button_label": localized_text(
            language,
            "Kampanya Analizini Aç",
            "Open Campaign Analysis",
        ),
        "key": "home_open_campaign",
    },
    {
        "title": localized_text(
            language,
            "Bütçe Optimizasyonu",
            "Budget Optimizer",
        ),
        "description": localized_text(
            language,
            (
                "Bütçe senaryolarını, kampanya önerilerini ve "
                "portföy dağılımını inceleyin."
            ),
            (
                "Review budget scenarios, campaign recommendations "
                "and portfolio allocation."
            ),
        ),
        "page_path": "pages/3_Budget_Optimizer.py",
        "button_label": localized_text(
            language,
            "Bütçe Optimizasyonunu Aç",
            "Open Budget Optimizer",
        ),
        "key": "home_open_budget",
    },
    {
        "title": localized_text(
            language,
            "AI Analizleri",
            "AI Insights",
        ),
        "description": localized_text(
            language,
            (
                "Fırsatları, riskleri, model performansını ve "
                "karar gerekçelerini inceleyin."
            ),
            (
                "Review opportunities, risks, model performance "
                "and decision rationale."
            ),
        ),
        "page_path": "pages/4_AI_Insights.py",
        "button_label": localized_text(
            language,
            "AI Analizlerini Aç",
            "Open AI Insights",
        ),
        "key": "home_open_insights",
    },
]

for column, page in zip(
    second_navigation_row,
    navigation_pages,
):
    with column:
        render_navigation_card(
            title=page["title"],
            description=page["description"],
            page_path=page["page_path"],
            button_label=page["button_label"],
            key=page["key"],
        )
        
render_read_only_footer(language)



