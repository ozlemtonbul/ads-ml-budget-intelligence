from __future__ import annotations

from datetime import date

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)


from config.settings import TARGET_ROAS
from dashboard_demo.components.cards import (
    render_data_coverage,
    render_kpi_cards,
)
from dashboard_demo.components.charts import (
    render_performance_charts,
)
from dashboard_demo.components.export import (
    render_export_buttons,
)
from dashboard_demo.components.tables import (
    hide_native_dataframe_toolbar,
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
)
from dashboard_demo.services.executive_metrics import (
    build_daily_trend,
    build_kpi_comparison,
    calculate_kpis,
)
from dashboard_demo.utils import (
    find_first_column,
    get_latest_output_time,
    load_csv,
)

CHANNEL_LABELS_TR = {
    "2": "Google Arama",
    "3": "Google Görüntülü Reklam Ağı",
    "6": "YouTube",
}

CHANNEL_LABELS_EN = {
    "2": "Google Search",
    "3": "Google Display Network",
    "6": "YouTube",
}


def channel_label(value: object, language: str) -> str:
    """Return a readable channel name while preserving its source code."""

    normalized = str(value)
    labels = (
        CHANNEL_LABELS_TR
        if language == "tr"
        else CHANNEL_LABELS_EN
    )
    name = labels.get(normalized)
    return (
        f"{name} ({normalized})"
        if name
        else normalized
    )


def dominant_dimension_by_spend(
    dataframe: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """Return each campaign's highest-spend dimension value."""

    if dataframe.empty:
        return pd.DataFrame(
            columns=["Campaign", dimension]
        )

    ranked = (
        dataframe[
            [
                "Campaign",
                dimension,
                "Spend",
            ]
        ]
        .groupby(
            [
                "Campaign",
                dimension,
            ],
            as_index=False,
            dropna=False,
        )["Spend"]
        .sum()
        .sort_values(
            [
                "Campaign",
                "Spend",
                dimension,
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="stable",
        )
    )

    return (
        ranked.drop_duplicates(
            subset=["Campaign"],
            keep="first",
        )[
            [
                "Campaign",
                dimension,
            ]
        ]
        .reset_index(drop=True)
    )


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return one numeric float column with safe missing values."""

    if column not in dataframe.columns:
        return pd.Series(
            0.0,
            index=dataframe.index,
            dtype="float64",
        )

    return (
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .fillna(0.0)
        .astype("float64")
    )


def apply_dimension_filters(
    dataframe: pd.DataFrame,
    campaigns: list[str],
    categories: list[str],
    channels: list[str],
) -> pd.DataFrame:
    """Apply the same campaign dimensions to any date period."""

    result = dataframe.copy()

    if campaigns and "Campaign" in result.columns:
        result = result[
            result["Campaign"].astype(str).isin(
                campaigns
            )
        ]

    if categories and "Category" in result.columns:
        result = result[
            result["Category"].astype(str).isin(
                categories
            )
        ]

    if channels and "Channel" in result.columns:
        channel_values = (
            result["Channel"]
            .astype(str)
        )

        result = result[
            channel_values.isin(channels)
        ]

    return result.reset_index(drop=True)


def aggregate_campaigns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build weighted campaign-level performance metrics."""

    if dataframe.empty:
        return pd.DataFrame()

    campaign_column = find_first_column(
        dataframe,
        ["Campaign", "CampaignName"],
    )

    if campaign_column is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        index=dataframe.index
    )

    working["Campaign"] = (
        dataframe[campaign_column]
        .fillna("UNKNOWN")
        .astype(str)
    )

    for dimension in [
        "Channel",
        "Category",
    ]:
        if dimension in dataframe.columns:
            working[dimension] = (
                dataframe[dimension]
                .fillna("UNKNOWN")
                .astype(str)
            )
        else:
            working[dimension] = "UNKNOWN"

    source_columns = {
        "Spend": ["Spend", "Cost", "AdSpend"],
        "Revenue": [
            "ConversionValue",
            "Revenue",
            "PurchaseRevenue",
        ],
        "Conversions": [
            "Conversions",
            "Purchases",
            "Transactions",
        ],
        "Profit": ["Profit"],
        "Clicks": ["Clicks"],
        "Impressions": ["Impressions"],
    }

    for target, candidates in source_columns.items():
        source = find_first_column(
            dataframe,
            candidates,
        )

        working[target] = (
            numeric_series(
                dataframe,
                source,
            )
            if source is not None
            else 0.0
        )

    result = (
        working.groupby(
            "Campaign",
            as_index=False,
            dropna=False,
        )
        .agg(
            Spend=("Spend", "sum"),
            Revenue=("Revenue", "sum"),
            Conversions=("Conversions", "sum"),
            Profit=("Profit", "sum"),
            Clicks=("Clicks", "sum"),
            Impressions=("Impressions", "sum"),
        )
    )

    for dimension in [
        "Channel",
        "Category",
    ]:
        result = result.merge(
            dominant_dimension_by_spend(
                working,
                dimension,
            ),
            on="Campaign",
            how="left",
            validate="one_to_one",
        )

    result["Channel"] = (
        result["Channel"]
        .fillna("UNKNOWN")
        .astype(str)
    )
    result["Category"] = (
        result["Category"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    result = result[
        [
            "Campaign",
            "Channel",
            "Category",
            "Spend",
            "Revenue",
            "Conversions",
            "Profit",
            "Clicks",
            "Impressions",
        ]
    ]

    result["ROAS"] = (
        result["Revenue"]
        .div(
            result["Spend"].where(
                result["Spend"] > 0
            )
        )
        .fillna(0.0)
    )

    result["CPA"] = (
        result["Spend"]
        .div(
            result["Conversions"].where(
                result["Conversions"] > 0
            )
        )
        .fillna(0.0)
    )

    result["ConversionRate"] = (
        result["Conversions"]
        .div(
            result["Clicks"].where(
                result["Clicks"] > 0
            )
        )
        .fillna(0.0)
    )

    result["CTR"] = (
        result["Clicks"]
        .div(
            result["Impressions"].where(
                result["Impressions"] > 0
            )
        )
        .fillna(0.0)
    )

    result["TargetROAS"] = TARGET_ROAS
    result["ROASGap"] = (
        result["ROAS"] - TARGET_ROAS
    )

    result["ROASStatus"] = "On Target"
    result.loc[
        result["ROAS"] < TARGET_ROAS * 0.9,
        "ROASStatus",
    ] = "Below Target"
    result.loc[
        result["ROAS"] > TARGET_ROAS * 1.1,
        "ROASStatus",
    ] = "Above Target"

    return (
        result.sort_values(
            "Revenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def add_campaign_comparison(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    comparison_available: bool,
) -> pd.DataFrame:
    """Attach campaign-level changes when coverage is sufficient."""

    result = current.copy()

    if result.empty or not comparison_available:
        return result

    previous_values = previous[
        [
            "Campaign",
            "Spend",
            "Revenue",
            "Conversions",
            "ROAS",
            "CPA",
        ]
    ].rename(
        columns={
            column: f"Previous{column}"
            for column in [
                "Spend",
                "Revenue",
                "Conversions",
                "ROAS",
                "CPA",
            ]
        }
    )

    result = result.merge(
        previous_values,
        on="Campaign",
        how="left",
    )

    for metric in [
        "Spend",
        "Revenue",
        "Conversions",
        "ROAS",
        "CPA",
    ]:
        previous_column = f"Previous{metric}"
        change_column = f"{metric}ChangePct"

        previous_metric = pd.to_numeric(
            result[previous_column],
            errors="coerce",
        )

        result[change_column] = (
            (
                pd.to_numeric(
                    result[metric],
                    errors="coerce",
                )
                - previous_metric
            )
            .div(
                previous_metric.abs().where(
                    previous_metric != 0
                )
            )
            .mul(100)
        )

    return result


def localized_campaign_table(
    dataframe: pd.DataFrame,
    language: str,
) -> pd.DataFrame:
    """Create a localized, manager-ready campaign table."""

    if dataframe.empty:
        return pd.DataFrame()

    selected_columns = [
        "Campaign",
        "Channel",
        "Category",
        "Spend",
        "Revenue",
        "Conversions",
        "ROAS",
        "CPA",
        "Profit",
        "ROASGap",
        "ROASStatus",
    ]

    optional_change_columns = [
        "RevenueChangePct",
        "SpendChangePct",
        "ROASChangePct",
    ]

    selected_columns.extend(
        column
        for column in optional_change_columns
        if column in dataframe.columns
    )

    table = dataframe[
        selected_columns
    ].copy()

    translations = (
        {
            "Campaign": "Kampanya",
            "Channel": "Kanal",
            "Category": "Kategori",
            "Spend": "Harcama",
            "Revenue": "Gelir",
            "Conversions": "Dönüşüm",
            "ROAS": "ROAS",
            "CPA": "CPA",
            "Profit": "Kâr",
            "ROASGap": "Hedef ROAS Farkı",
            "ROASStatus": "ROAS Durumu",
            "RevenueChangePct": "Gelir Değişimi (%)",
            "SpendChangePct": "Harcama Değişimi (%)",
            "ROASChangePct": "ROAS Değişimi (%)",
        }
        if language == "tr"
        else {
            "Campaign": "Campaign",
            "Channel": "Channel",
            "Category": "Category",
            "Spend": "Spend",
            "Revenue": "Revenue",
            "Conversions": "Conversions",
            "ROAS": "ROAS",
            "CPA": "CPA",
            "Profit": "Profit",
            "ROASGap": "Target ROAS Gap",
            "ROASStatus": "ROAS Status",
            "RevenueChangePct": "Revenue Change (%)",
            "SpendChangePct": "Spend Change (%)",
            "ROASChangePct": "ROAS Change (%)",
        }
    )

    table = table.rename(
        columns=translations
    )

    channel_column = (
        "Kanal"
        if language == "tr"
        else "Channel"
    )
    category_column = (
        "Kategori"
        if language == "tr"
        else "Category"
    )

    table[channel_column] = table[channel_column].map(
        lambda value: channel_label(value, language)
    )
    table[category_column] = table[category_column].replace(
        {
            "MULTIPLE": (
                "Birden Fazla"
                if language == "tr"
                else "Multiple"
            ),
            "UNKNOWN": (
                "Bilinmiyor"
                if language == "tr"
                else "Unknown"
            ),
        }
    )

    status_map = (
        {
            "Above Target": "Hedef Üstü",
            "On Target": "Hedefte",
            "Below Target": "Hedef Altı",
        }
        if language == "tr"
        else {}
    )

    status_column = (
        "ROAS Durumu"
        if language == "tr"
        else "ROAS Status"
    )

    table[status_column] = (
        table[status_column]
        .replace(status_map)
    )

    return table


def campaign_table_column_config(
    language: str,
) -> dict[str, object]:
    """Return readable formats without changing numeric values."""

    labels = (
        {
            "spend": "Harcama",
            "revenue": "Gelir",
            "conversions": "Dönüşüm",
            "profit": "Kâr",
            "gap": "Hedef ROAS Farkı",
            "revenue_change": "Gelir Değişimi (%)",
            "spend_change": "Harcama Değişimi (%)",
            "roas_change": "ROAS Değişimi (%)",
        }
        if language == "tr"
        else {
            "spend": "Spend",
            "revenue": "Revenue",
            "conversions": "Conversions",
            "profit": "Profit",
            "gap": "Target ROAS Gap",
            "revenue_change": "Revenue Change (%)",
            "spend_change": "Spend Change (%)",
            "roas_change": "ROAS Change (%)",
        }
    )

    return {
        labels["spend"]: st.column_config.NumberColumn(
            labels["spend"],
            format="₺%.2f",
        ),
        labels["revenue"]: st.column_config.NumberColumn(
            labels["revenue"],
            format="₺%.2f",
        ),
        labels["conversions"]: st.column_config.NumberColumn(
            labels["conversions"],
            format="%.2f",
        ),
        "ROAS": st.column_config.NumberColumn(
            "ROAS",
            format="%.2fx",
        ),
        "CPA": st.column_config.NumberColumn(
            "CPA",
            format="₺%.2f",
        ),
        labels["profit"]: st.column_config.NumberColumn(
            labels["profit"],
            format="₺%.2f",
        ),
        labels["gap"]: st.column_config.NumberColumn(
            labels["gap"],
            format="%+.2fx",
        ),
        labels["revenue_change"]: (
            st.column_config.NumberColumn(
                labels["revenue_change"],
                format="%+.1f%%",
            )
        ),
        labels["spend_change"]: (
            st.column_config.NumberColumn(
                labels["spend_change"],
                format="%+.1f%%",
            )
        ),
        labels["roas_change"]: (
            st.column_config.NumberColumn(
                labels["roas_change"],
                format="%+.1f%%",
            )
        ),
    }


def render_campaign_bar_chart(
    campaign_data: pd.DataFrame,
    metric: str,
    language: str,
) -> None:
    """Render a top-campaign horizontal bar chart."""

    if campaign_data.empty:
        return

    metric_labels = {
        "Revenue": localized_text(
            language,
            "Gelir",
            "Revenue",
        ),
        "Spend": localized_text(
            language,
            "Harcama",
            "Spend",
        ),
        "Profit": localized_text(
            language,
            "Kâr",
            "Profit",
        ),
        "ROAS": "ROAS",
        "Conversions": localized_text(
            language,
            "Dönüşüm",
            "Conversions",
        ),
    }

    top_campaigns = (
        campaign_data.sort_values(
            metric,
            ascending=False,
        )
        .head(12)
        .sort_values(
            metric,
            ascending=True,
        )
    )

    colors = [
        (
            "#22c55e"
            if roas >= TARGET_ROAS
            else "#ef4444"
        )
        for roas in top_campaigns["ROAS"]
    ]

    figure = go.Figure(
        go.Bar(
            x=top_campaigns[metric],
            y=top_campaigns["Campaign"],
            orientation="h",
            marker_color=colors,
            customdata=top_campaigns[
                [
                    "Revenue",
                    "Spend",
                    "ROAS",
                    "Conversions",
                ]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Revenue: ₺%{customdata[0]:,.2f}<br>"
                "Spend: ₺%{customdata[1]:,.2f}<br>"
                "ROAS: %{customdata[2]:.2f}x<br>"
                "Conversions: %{customdata[3]:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=520,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        xaxis_title=metric_labels[metric],
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        key="campaign_analysis_bar_chart",
    )


# ---------------------------------------------------------
# LOAD AND INITIALIZE
# ---------------------------------------------------------

daily_df = load_csv(
    "ads_daily_fact.csv"
)

available_start, available_end = (
    get_available_date_bounds(
        daily_df
    )
)

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=(
        "Kampanya Analizi"
        if initial_language == "tr"
        else "Campaign Analysis"
    ),
    page_icon="🎯",
    title=translate(
        "campaign_analysis",
        initial_language,
    ),
    subtitle=translate(
        "campaign_analysis_description",
        initial_language,
    ),
    reference_date=date(2026, 1, 31),
)

language = context.language
filters = context.filters

# The page already provides reliable CSV and Excel exports.
# Hide Streamlit's secondary table toolbar to avoid duplicate,
# browser-dependent CSV download controls.
hide_native_dataframe_toolbar()


# ---------------------------------------------------------
# DATE FILTERS
# ---------------------------------------------------------

current_period_df = filter_by_date(
    daily_df,
    filters.start_date,
    filters.end_date,
)

comparison_enabled = (
    filters.comparison != "no_comparison"
    and filters.comparison_start_date is not None
    and filters.comparison_end_date is not None
)

if comparison_enabled:
    comparison_period_df = filter_by_date(
        daily_df,
        filters.comparison_start_date,
        filters.comparison_end_date,
    )
else:
    comparison_period_df = pd.DataFrame()


# ---------------------------------------------------------
# CAMPAIGN DIMENSION FILTERS
# ---------------------------------------------------------

st.subheader(
    localized_text(
        language,
        "Kampanya Filtreleri",
        "Campaign Filters",
    )
)

filter_columns = st.columns(3)

campaign_options = (
    sorted(
        daily_df["Campaign"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "Campaign" in daily_df.columns
    else []
)

category_options = (
    sorted(
        daily_df["Category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "Category" in daily_df.columns
    else []
)

channel_options = (
    sorted(
        daily_df["Channel"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if "Channel" in daily_df.columns
    else []
)

selected_campaigns = (
    filter_columns[0].multiselect(
        localized_text(
            language,
            "Kampanya",
            "Campaign",
        ),
        campaign_options,
        key="campaign_analysis_campaigns",
        placeholder=localized_text(
            language,
            "Tüm kampanyalar",
            "All campaigns",
        ),
    )
)

selected_categories = (
    filter_columns[1].multiselect(
        localized_text(
            language,
            "Kategori",
            "Category",
        ),
        category_options,
        key="campaign_analysis_categories",
        placeholder=localized_text(
            language,
            "Tüm kategoriler",
            "All categories",
        ),
    )
)

selected_channels = (
    filter_columns[2].multiselect(
        localized_text(
            language,
            "Kanal",
            "Channel",
        ),
        channel_options,
        format_func=lambda value: channel_label(
            value,
            language,
        ),
        key="campaign_analysis_channels",
        placeholder=localized_text(
            language,
            "Tüm kanallar",
            "All channels",
        ),
    )
)

current_df = apply_dimension_filters(
    current_period_df,
    selected_campaigns,
    selected_categories,
    selected_channels,
)

comparison_df = apply_dimension_filters(
    comparison_period_df,
    selected_campaigns,
    selected_categories,
    selected_channels,
)


# ---------------------------------------------------------
# COVERAGE AND KPI
# ---------------------------------------------------------

current_coverage = calculate_date_coverage(
    source_dataframe=daily_df,
    filtered_dataframe=current_df,
    start_date=filters.start_date,
    end_date=filters.end_date,
)

if comparison_enabled:
    comparison_coverage = calculate_date_coverage(
        source_dataframe=daily_df,
        filtered_dataframe=comparison_df,
        start_date=filters.comparison_start_date,
        end_date=filters.comparison_end_date,
    )
else:
    comparison_coverage = DateCoverage(
        selected_days=0,
        available_days=0,
        coverage_ratio=0.0,
        available_start=available_start,
        available_end=available_end,
    )

render_data_coverage(
    current_coverage=current_coverage,
    comparison_coverage=comparison_coverage,
    comparison_enabled=comparison_enabled,
    data_age_days=calculate_data_age_days(
        available_end
    ),
    language=language,
)

if current_df.empty:
    st.error(
        translate(
            "no_data",
            language,
        )
    )
    render_read_only_footer(language)
    st.stop()

current_kpis = calculate_kpis(
    current_df
)

previous_kpis = calculate_kpis(
    comparison_df
)

kpi_comparison = build_kpi_comparison(
    current=current_kpis,
    previous=previous_kpis,
    comparison_coverage=comparison_coverage,
)

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


# ---------------------------------------------------------
# CAMPAIGN TABLE AND COMPARISON
# ---------------------------------------------------------

campaign_summary = aggregate_campaigns(
    current_df
)

previous_campaign_summary = aggregate_campaigns(
    comparison_df
)

campaign_summary = add_campaign_comparison(
    current=campaign_summary,
    previous=previous_campaign_summary,
    comparison_available=(
        kpi_comparison.is_available
    ),
)

display_table = localized_campaign_table(
    campaign_summary,
    language,
)

st.divider()
st.subheader(
    localized_text(
        language,
        "Kampanya Performans Tablosu",
        "Campaign Performance Table",
    )
)

st.dataframe(
    display_table,
    width="stretch",
    hide_index=True,
    column_config=campaign_table_column_config(
        language
    ),
    key="campaign_analysis_performance_table",
)


# ---------------------------------------------------------
# TOP CAMPAIGN CHART
# ---------------------------------------------------------

chart_metric_labels = {
    localized_text(
        language,
        "Gelir",
        "Revenue",
    ): "Revenue",
    localized_text(
        language,
        "Harcama",
        "Spend",
    ): "Spend",
    localized_text(
        language,
        "Kâr",
        "Profit",
    ): "Profit",
    "ROAS": "ROAS",
    localized_text(
        language,
        "Dönüşüm",
        "Conversions",
    ): "Conversions",
}

selected_metric_label = st.selectbox(
    localized_text(
        language,
        "Grafik Metriği",
        "Chart Metric",
    ),
    options=list(
        chart_metric_labels.keys()
    ),
    key="campaign_analysis_chart_metric",
)

render_campaign_bar_chart(
    campaign_data=campaign_summary,
    metric=chart_metric_labels[
        selected_metric_label
    ],
    language=language,
)


# ---------------------------------------------------------
# DAILY TREND
# ---------------------------------------------------------

st.divider()

render_performance_charts(
    trend=build_daily_trend(
        current_df
    ),
    target_roas=TARGET_ROAS,
    language=language,
)


# ---------------------------------------------------------
# OPPORTUNITY AND RISK
# ---------------------------------------------------------

st.divider()
st.subheader(
    localized_text(
        language,
        "Kampanya Karar Sinyalleri",
        "Campaign Decision Signals",
    )
)

active_campaigns = campaign_summary[
    campaign_summary["Spend"] > 0
].copy()

opportunities = (
    active_campaigns[
        active_campaigns["ROAS"] >= TARGET_ROAS
    ]
    .sort_values(
        [
            "Profit",
            "Revenue",
            "ROAS",
        ],
        ascending=False,
    )
    .head(5)
)

risks = (
    active_campaigns[
        active_campaigns["ROAS"] < TARGET_ROAS
    ]
    .sort_values(
        [
            "ROASGap",
            "Spend",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .head(5)
)

opportunity_column, risk_column = st.columns(2)

with opportunity_column:
    st.markdown(
        localized_text(
            language,
            "**İlk 5 Fırsat**",
            "**Top 5 Opportunities**",
        )
    )

    st.dataframe(
        localized_campaign_table(
            opportunities,
            language,
        ),
        width="stretch",
        hide_index=True,
        column_config=campaign_table_column_config(
            language
        ),
        key="campaign_analysis_opportunities",
    )

with risk_column:
    st.markdown(
        localized_text(
            language,
            "**İlk 5 Risk**",
            "**Top 5 Risks**",
        )
    )

    if risks.empty:
        st.success(
            localized_text(
                language,
                (
                    "Seçilen dönemde hedef ROAS altında "
                    "aktif kampanya bulunmuyor."
                ),
                (
                    "No active campaign is below target ROAS "
                    "for the selected period."
                ),
            )
        )
    else:
        st.dataframe(
            localized_campaign_table(
                risks,
                language,
            ),
            width="stretch",
            hide_index=True,
            column_config=campaign_table_column_config(
                language
            ),
            key="campaign_analysis_risks",
        )


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

st.divider()
st.subheader(
    localized_text(
        language,
        "Kampanya Raporunu Dışa Aktar",
        "Export Campaign Report",
    )
)

render_export_buttons(
    csv_dataframe=display_table,
    excel_sheets={
        "Campaign Summary": display_table,
        "Daily Data": current_df,
        "Opportunities": localized_campaign_table(
            opportunities,
            language,
        ),
        "Risks": localized_campaign_table(
            risks,
            language,
        ),
    },
    file_name="campaign_analysis_report",
    language=language,
    key_prefix="campaign_analysis",
)

render_read_only_footer(language)






