from __future__ import annotations

from datetime import date

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(
    0,
    PROJECT_ROOT_STR,
)


from config.settings import TARGET_ROAS

from dashboard_demo.components.cards import (
    render_action_cards,
    render_ai_executive_summary,
    render_data_coverage,
    render_kpi_cards,
)
from dashboard_demo.components.charts import (
    render_performance_charts,
    render_portfolio_chart,
)
from dashboard_demo.components.export import (
    render_export_buttons,
)
from dashboard_demo.components.tables import (
    render_detailed_data,
    render_opportunity_risk_tables,
    render_optimization_table,
)
from dashboard_demo.i18n import translate
from dashboard_demo.layout_demo import (
    initialize_dashboard,
    localized_text,
    render_read_only_footer,
)
from dashboard_demo.services.executive_data import (
    DateCoverage,
    calculate_data_age_days,
    calculate_date_coverage,
    filter_by_date,
    get_available_date_bounds,
    get_recommendation_period,
    load_executive_data,
    recommendation_period_is_known,
)
from dashboard_demo.services.executive_metrics import (
    build_daily_trend,
    build_kpi_comparison,
    calculate_kpis,
    calculate_model_r2,
)
from dashboard_demo.services.executive_scoring import (
    build_action_summary,
    build_display_table,
    enrich_recommendations,
    get_top_opportunities,
    get_top_risks,
)
from dashboard_demo.utils import get_latest_output_time


# ---------------------------------------------------------
# LOAD SOURCE DATA
# ---------------------------------------------------------
data = load_executive_data()

available_start, available_end = (
    get_available_date_bounds(
        data.daily
    )
)


# ---------------------------------------------------------
# INITIALIZE DASHBOARD
# ---------------------------------------------------------
initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=(
        "Yönetici Özeti"
        if initial_language == "tr"
        else "Executive Overview"
    ),
    page_icon="📈",
    title=translate(
        "executive_overview",
        initial_language,
    ),
    subtitle=translate(
        "executive_overview_description",
        initial_language,
    ),
    reference_date=date(2026, 1, 31),
)

language = context.language
filters = context.filters


# ---------------------------------------------------------
# FILTER CURRENT PERIOD
# ---------------------------------------------------------
current_df = filter_by_date(
    data.daily,
    filters.start_date,
    filters.end_date,
)

current_coverage = calculate_date_coverage(
    source_dataframe=data.daily,
    filtered_dataframe=current_df,
    start_date=filters.start_date,
    end_date=filters.end_date,
)


# ---------------------------------------------------------
# FILTER COMPARISON PERIOD
# ---------------------------------------------------------
comparison_enabled = (
    filters.comparison != "no_comparison"
    and filters.comparison_start_date is not None
    and filters.comparison_end_date is not None
)

if comparison_enabled:
    comparison_df = filter_by_date(
        data.daily,
        filters.comparison_start_date,
        filters.comparison_end_date,
    )

    comparison_coverage = (
        calculate_date_coverage(
            source_dataframe=data.daily,
            filtered_dataframe=comparison_df,
            start_date=(
                filters.comparison_start_date
            ),
            end_date=(
                filters.comparison_end_date
            ),
        )
    )
else:
    comparison_df = pd.DataFrame()

    comparison_coverage = DateCoverage(
        selected_days=0,
        available_days=0,
        coverage_ratio=0.0,
        available_start=available_start,
        available_end=available_end,
    )


# ---------------------------------------------------------
# CALCULATE KPIs AND COMPARISON
# ---------------------------------------------------------
current_kpis = calculate_kpis(
    current_df
)

comparison_kpis = calculate_kpis(
    comparison_df
)

kpi_comparison = build_kpi_comparison(
    current=current_kpis,
    previous=comparison_kpis,
    comparison_coverage=comparison_coverage,
)

trend_df = build_daily_trend(
    current_df
)


# ---------------------------------------------------------
# PREPARE RECOMMENDATIONS
# ---------------------------------------------------------
enriched_recommendations = (
    enrich_recommendations(
        data.recommendations,
        target_roas=TARGET_ROAS,
    )
)

display_recommendations = (
    build_display_table(
        enriched_recommendations,
        language,
    )
)

action_summary = build_action_summary(
    enriched_recommendations
)

top_opportunities = get_top_opportunities(
    enriched_recommendations,
    limit=3,
)

top_risks = get_top_risks(
    enriched_recommendations,
    limit=3,
)

model_r2 = calculate_model_r2(
    data.model_metrics
)

data_age_days = calculate_data_age_days(
    available_end
)

recommendation_start, recommendation_end = (
    get_recommendation_period(
        data.recommendations
    )
)

recommendation_period_known = (
    recommendation_period_is_known(
        data.recommendations
    )
    and recommendation_start is not None
    and recommendation_end is not None
)

recommendation_matches_selection = (
    recommendation_period_known
    and recommendation_start == filters.start_date
    and recommendation_end == filters.end_date
)

if not recommendation_matches_selection:
    enriched_recommendations = enriched_recommendations.iloc[0:0].copy()
    display_recommendations = display_recommendations.iloc[0:0].copy()
    action_summary = build_action_summary(
        enriched_recommendations
    )
    top_opportunities = enriched_recommendations.copy()
    top_risks = enriched_recommendations.copy()
    model_r2 = None


# ---------------------------------------------------------
# PLATFORM STATUS
# ---------------------------------------------------------
status_column, refresh_column = st.columns(
    [4, 1]
)

with status_column:
    st.success(
        translate(
            "platform_online",
            language,
        ),
        icon="✅",
    )

with refresh_column:
    if st.button(
        translate(
            "refresh_data",
            language,
        ),
        width="stretch",
        key="executive_refresh_data",
    ):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------
# DATA COVERAGE
# ---------------------------------------------------------
render_data_coverage(
    current_coverage=current_coverage,
    comparison_coverage=comparison_coverage,
    comparison_enabled=comparison_enabled,
    data_age_days=data_age_days,
    language=language,
)

if not recommendation_period_known:
    st.warning(
        localized_text(
            language,
            (
                "Optimizasyon çıktısında analiz dönemi bilgisi "
                "bulunmuyor. KPI ve grafikler seçilen tarihe göre "
                "hesaplanır; bütçe önerileri ise son pipeline "
                "çalıştırmasının statik çıktılarıdır. Pipeline yeniden "
                "çalıştırıldığında dönem bilgisi otomatik eklenecektir."
            ),
            (
                "The optimization output does not contain analysis "
                "period metadata. KPIs and charts use the selected "
                "dates, while budget recommendations are static "
                "results from the latest pipeline run. Period metadata "
                "will be added automatically after the pipeline reruns."
            ),
        ),
        icon="⚠️",
    )
elif not recommendation_matches_selection:
    st.warning(
        localized_text(
            language,
            (
                "Seçilen KPI dönemi ile optimizasyon dönemi farklı. "
                f"Seçilen dönem: {filters.start_date:%d.%m.%Y} – "
                f"{filters.end_date:%d.%m.%Y}. "
                "Öneri dönemi: "
                f"{recommendation_start:%d.%m.%Y} – "
                f"{recommendation_end:%d.%m.%Y}. "
                "Eski öneriler gösterilmiyor. Tarih bölümünü açıp "
                "“Seçilen Dönemi Analiz Et” düğmesine basın."
            ),
            (
                "The selected KPI period differs from the optimization "
                f"period. Selected: {filters.start_date:%Y-%m-%d} – "
                f"{filters.end_date:%Y-%m-%d}. "
                "Recommendations: "
                f"{recommendation_start:%Y-%m-%d} – "
                f"{recommendation_end:%Y-%m-%d}. "
                "Stale recommendations are hidden. Open the Date section "
                "and click “Analyze Selected Period”."
            ),
        ),
        icon="⚠️",
    )
else:
    st.success(
        localized_text(
            language,
            (
                "KPI dönemi ile optimizasyon önerisi dönemi uyumlu."
            ),
            (
                "The KPI period and optimization recommendation "
                "period match."
            ),
        ),
        icon="✅",
    )


# ---------------------------------------------------------
# STOP SAFELY WHEN CURRENT DATA IS EMPTY
# ---------------------------------------------------------
if current_df.empty:
    st.error(
        translate(
            "no_data",
            language,
        )
    )

    st.caption(
        f"{translate('last_update', language)}: "
        f"{get_latest_output_time()}"
    )

    render_read_only_footer(
        language
    )

    st.stop()


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
render_kpi_cards(
    current=current_kpis,
    comparison=kpi_comparison,
    target_roas=TARGET_ROAS,
    language=language,
)

st.caption(
    f"{translate('last_update', language)}: "
    f"{get_latest_output_time()}"
)

st.divider()


# ---------------------------------------------------------
# PERFORMANCE CHARTS
# ---------------------------------------------------------
render_performance_charts(
    trend=trend_df,
    target_roas=TARGET_ROAS,
    language=language,
)

st.divider()


# ---------------------------------------------------------
# AI EXECUTIVE SUMMARY
# ---------------------------------------------------------
render_ai_executive_summary(
    current_kpis=current_kpis,
    action_summary=action_summary,
    opportunities=top_opportunities,
    risks=top_risks,
    target_roas=TARGET_ROAS,
    executive_commentary=(
        data.executive_commentary
    ),
    language=language,
)

st.divider()


# ---------------------------------------------------------
# OPTIMIZATION TABLE
# ---------------------------------------------------------
render_optimization_table(
    enriched_recommendations=(
        enriched_recommendations
    ),
    language=language,
    limit=10,
)


# ---------------------------------------------------------
# PORTFOLIO CHART
# ---------------------------------------------------------
render_portfolio_chart(
    enriched_recommendations=(
        enriched_recommendations
    ),
    language=language,
    limit=8,
)

st.divider()


# ---------------------------------------------------------
# ACTION CENTER
# ---------------------------------------------------------
render_action_cards(
    summary=action_summary,
    model_r2=model_r2,
    language=language,
)


# ---------------------------------------------------------
# TOP OPPORTUNITIES AND RISKS
# ---------------------------------------------------------
render_opportunity_risk_tables(
    opportunities=top_opportunities,
    risks=top_risks,
    language=language,
)


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------
st.subheader(
    localized_text(
        language,
        "Raporu Dışa Aktar",
        "Export Report",
    )
)

render_export_buttons(
    csv_dataframe=display_recommendations,
    excel_sheets={
        "Recommendations": (
            display_recommendations
        ),
        "Portfolio": data.portfolio,
        "Daily Data": current_df,
        "Model Metrics": data.model_metrics,
        "Recommendation Summary": (
            data.recommendation_summary
        ),
    },
    file_name="executive_decision_report",
    language=language,
    key_prefix="executive",
)

if st.button(
    localized_text(
        language,
        "AI Asistanına Sor",
        "Ask AI Assistant",
    ),
    width="stretch",
    key="executive_ask_ai",
):
    st.switch_page(
        "pages/5_Ask_AI.py"
    )


# ---------------------------------------------------------
# DETAILED DATA
# ---------------------------------------------------------
render_detailed_data(
    recommendation_summary=(
        data.recommendation_summary
    ),
    portfolio=data.portfolio,
    daily=current_df,
    language=language,
)


# ---------------------------------------------------------
# READ-ONLY NOTICE
# ---------------------------------------------------------
render_read_only_footer(
    language
)






