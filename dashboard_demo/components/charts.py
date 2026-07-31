from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard_demo.layout import localized_text


REVENUE_COLOR = "#60a5fa"
SPEND_COLOR = "#0f4c81"
ROAS_COLOR = "#4ade80"
TARGET_COLOR = "#ef4444"
MA7_COLOR = "#fca5a5"
MA30_COLOR = "#fbbf24"


def render_performance_charts(
    trend: pd.DataFrame,
    target_roas: float,
    language: str,
) -> None:
    """
    Render revenue/spend and ROAS performance charts.
    """

    st.subheader(
        localized_text(
            language,
            "Performans Trendi",
            "Performance Trend",
        )
    )

    available_days = (
        trend["Date"].nunique()
        if not trend.empty and "Date" in trend.columns
        else 0
    )

    toggle_1, toggle_2 = st.columns(2)

    show_ma_7 = toggle_1.toggle(
        localized_text(
            language,
            "7 Günlük Ortalama",
            "7-Day Average",
        ),
        value=available_days >= 7,
        disabled=available_days < 7,
        key="executive_show_ma_7",
    )

    show_ma_30 = toggle_2.toggle(
        localized_text(
            language,
            "30 Günlük Ortalama",
            "30-Day Average",
        ),
        value=False,
        disabled=available_days < 30,
        key="executive_show_ma_30",
    )

    if trend.empty:
        st.info(
            localized_text(
                language,
                "Grafik için veri bulunmuyor.",
                "No data is available for the chart.",
            )
        )
        return

    if available_days < 7:
        st.info(
            localized_text(
                language,
                (
                    "Hareketli ortalamalar için en az 7 günlük "
                    "veri seçin. Tek günlük seçimde yalnızca günlük "
                    "değer gösterilir."
                ),
                (
                    "Select at least 7 days for moving averages. "
                    "A one-day selection shows only the daily value."
                ),
            )
        )

    # ---------------------------------------------------------
    # REVENUE AND SPEND
    # ---------------------------------------------------------
    performance_figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    performance_figure.add_trace(
        go.Bar(
            x=trend["Date"],
            y=trend["Spend"],
            name=localized_text(
                language,
                "Reklam Harcaması",
                "Ad Spend",
            ),
            marker_color=SPEND_COLOR,
            opacity=0.65,
            hovertemplate=(
                "%{x|%d.%m.%Y}<br>"
                + localized_text(
                    language,
                    "Harcama",
                    "Spend",
                )
                + ": ₺%{y:,.2f}"
                + "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    performance_figure.add_trace(
        go.Scatter(
            x=trend["Date"],
            y=trend["Revenue"],
            name=localized_text(
                language,
                "Gelir",
                "Revenue",
            ),
            mode="lines+markers",
            line={
                "color": REVENUE_COLOR,
                "width": 2.5,
            },
            marker={
                "size": 6,
            },
            hovertemplate=(
                "%{x|%d.%m.%Y}<br>"
                + localized_text(
                    language,
                    "Gelir",
                    "Revenue",
                )
                + ": ₺%{y:,.2f}"
                + "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    if (
        show_ma_7
        and "RevenueMA7" in trend.columns
    ):
        performance_figure.add_trace(
            go.Scatter(
                x=trend["Date"],
                y=trend["RevenueMA7"],
                name=localized_text(
                    language,
                    "Gelir HO7",
                    "Revenue MA7",
                ),
                mode="lines",
                line={
                    "color": MA7_COLOR,
                    "width": 2,
                    "dash": "dash",
                },
                hovertemplate=(
                    "%{x|%d.%m.%Y}<br>"
                    "MA7: ₺%{y:,.2f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    if (
        show_ma_30
        and "RevenueMA30" in trend.columns
    ):
        performance_figure.add_trace(
            go.Scatter(
                x=trend["Date"],
                y=trend["RevenueMA30"],
                name=localized_text(
                    language,
                    "Gelir HO30",
                    "Revenue MA30",
                ),
                mode="lines",
                line={
                    "color": MA30_COLOR,
                    "width": 2,
                    "dash": "dot",
                },
                hovertemplate=(
                    "%{x|%d.%m.%Y}<br>"
                    "MA30: ₺%{y:,.2f}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    performance_figure.update_yaxes(
        title_text=localized_text(
            language,
            "Gelir",
            "Revenue",
        ),
        tickprefix="₺",
        tickformat="~s",
        secondary_y=False,
    )

    performance_figure.update_yaxes(
        title_text=localized_text(
            language,
            "Reklam Harcaması",
            "Ad Spend",
        ),
        tickprefix="₺",
        tickformat="~s",
        rangemode="tozero",
        secondary_y=True,
    )

    performance_figure.update_xaxes(
        title_text="",
        tickformat="%d %b",
        showgrid=False,
        nticks=max(1, min(8, available_days)),
    )

    performance_figure.update_layout(
        height=470,
        hovermode="x unified",
        barmode="overlay",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
    )

    st.plotly_chart(
        performance_figure,
        width="stretch",
        key="executive_performance_chart",
    )

    # ---------------------------------------------------------
    # ROAS AND TARGET
    # ---------------------------------------------------------
    st.subheader(
        localized_text(
            language,
            "ROAS ve Hedef Performansı",
            "ROAS and Target Performance",
        )
    )

    roas_figure = go.Figure()

    roas_marker_colors = [
        (
            ROAS_COLOR
            if value >= target_roas
            else TARGET_COLOR
        )
        for value in trend["ROAS"]
    ]

    roas_figure.add_trace(
        go.Scatter(
            x=trend["Date"],
            y=trend["ROAS"],
            name="ROAS",
            mode="lines+markers",
            line={
                "color": REVENUE_COLOR,
                "width": 2,
            },
            marker={
                "size": 8,
                "color": roas_marker_colors,
            },
            hovertemplate=(
                "%{x|%d.%m.%Y}<br>"
                "ROAS: %{y:.2f}x"
                "<extra></extra>"
            ),
        )
    )

    if (
        show_ma_7
        and "ROASMA7" in trend.columns
    ):
        roas_figure.add_trace(
            go.Scatter(
                x=trend["Date"],
                y=trend["ROASMA7"],
                name=localized_text(
                    language,
                    "ROAS HO7",
                    "ROAS MA7",
                ),
                mode="lines",
                line={
                    "color": "#1687f8",
                    "width": 2,
                    "dash": "dash",
                },
                hovertemplate=(
                    "%{x|%d.%m.%Y}<br>"
                    "ROAS MA7: %{y:.2f}x"
                    "<extra></extra>"
                ),
            )
        )

    if (
        show_ma_30
        and "ROASMA30" in trend.columns
    ):
        roas_figure.add_trace(
            go.Scatter(
                x=trend["Date"],
                y=trend["ROASMA30"],
                name=localized_text(
                    language,
                    "ROAS HO30",
                    "ROAS MA30",
                ),
                mode="lines",
                line={
                    "color": MA30_COLOR,
                    "width": 2,
                    "dash": "dot",
                },
                hovertemplate=(
                    "%{x|%d.%m.%Y}<br>"
                    "ROAS MA30: %{y:.2f}x"
                    "<extra></extra>"
                ),
            )
        )

    roas_figure.add_hline(
        y=target_roas,
        line_dash="dash",
        line_color=TARGET_COLOR,
        line_width=2,
        annotation_text=localized_text(
            language,
            "Hedef ROAS",
            "Target ROAS",
        ),
        annotation_position="top right",
    )

    roas_figure.update_yaxes(
        title_text="ROAS",
        ticksuffix="x",
        rangemode="tozero",
    )

    roas_figure.update_xaxes(
        title_text="",
        tickformat="%d %b",
        showgrid=False,
        nticks=max(1, min(8, available_days)),
    )

    roas_figure.update_layout(
        height=390,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20,
        },
    )

    st.plotly_chart(
        roas_figure,
        width="stretch",
        key="executive_roas_chart",
    )


def _shorten_campaign_name(
    value: object,
    maximum_length: int = 35,
) -> str:
    """Shorten long campaign names for chart axes."""

    text = str(value)

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3]
        + "..."
    )


def render_portfolio_chart(
    enriched_recommendations: pd.DataFrame,
    language: str,
    limit: int = 8,
) -> None:
    """
    Compare current spend and recommended budget.
    """

    st.subheader(
        localized_text(
            language,
            "Portföy Bütçe Dağılımı",
            "Portfolio Budget Allocation",
        )
    )

    if enriched_recommendations.empty:
        st.info(
            localized_text(
                language,
                "Portföy verisi bulunmuyor.",
                "Portfolio data is not available.",
            )
        )
        return

    required_columns = {
        "CampaignCanonical",
        "CurrentSpendCanonical",
        "RecommendedBudgetCanonical",
    }

    if not required_columns.issubset(
        enriched_recommendations.columns
    ):
        st.info(
            localized_text(
                language,
                "Portföy grafiği için gerekli alanlar eksik.",
                "Required portfolio fields are missing.",
            )
        )
        return

    portfolio = (
        enriched_recommendations[
            [
                "CampaignCanonical",
                "CurrentSpendCanonical",
                "RecommendedBudgetCanonical",
            ]
        ]
        .copy()
        .sort_values(
            "RecommendedBudgetCanonical",
            ascending=False,
        )
        .head(limit)
        .sort_values(
            "RecommendedBudgetCanonical",
            ascending=True,
        )
    )

    portfolio["CampaignShort"] = (
        portfolio["CampaignCanonical"].map(
            _shorten_campaign_name
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            y=portfolio["CampaignShort"],
            x=portfolio["CurrentSpendCanonical"],
            name=localized_text(
                language,
                "Mevcut Harcama",
                "Current Spend",
            ),
            orientation="h",
            marker_color="#7dd3fc",
            text=portfolio[
                "CurrentSpendCanonical"
            ].map(
                lambda value: f"₺{value:,.0f}"
            ),
            textposition="outside",
            customdata=portfolio[
                "CampaignCanonical"
            ],
            hovertemplate=(
                "%{customdata}<br>"
                + localized_text(
                    language,
                    "Mevcut Harcama",
                    "Current Spend",
                )
                + ": ₺%{x:,.2f}"
                + "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            y=portfolio["CampaignShort"],
            x=portfolio[
                "RecommendedBudgetCanonical"
            ],
            name=localized_text(
                language,
                "Önerilen Bütçe",
                "Recommended Budget",
            ),
            orientation="h",
            marker_color="#087ccc",
            text=portfolio[
                "RecommendedBudgetCanonical"
            ].map(
                lambda value: f"₺{value:,.0f}"
            ),
            textposition="outside",
            customdata=portfolio[
                "CampaignCanonical"
            ],
            hovertemplate=(
                "%{customdata}<br>"
                + localized_text(
                    language,
                    "Önerilen Bütçe",
                    "Recommended Budget",
                )
                + ": ₺%{x:,.2f}"
                + "<extra></extra>"
            ),
        )
    )

    figure.update_xaxes(
        title_text=localized_text(
            language,
            "Bütçe (₺)",
            "Budget (₺)",
        ),
        tickprefix="₺",
        rangemode="tozero",
    )

    figure.update_yaxes(
        title_text="",
        automargin=True,
    )

    figure.update_layout(
        height=max(
            430,
            58 * len(portfolio),
        ),
        barmode="group",
        bargap=0.25,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 20,
            "r": 80,
            "t": 70,
            "b": 30,
        },
    )

    st.plotly_chart(
        figure,
        width="stretch",
        key="executive_portfolio_chart",
    )



