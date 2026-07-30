from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.export import (
    render_export_buttons,
)
from dashboard.layout import localized_text
from dashboard.services.executive_scoring import (
    build_display_table,
)


def hide_native_dataframe_toolbar() -> None:
    """
    Hide Streamlit's native table toolbar.

    The native download button only exports CSV.
    Custom CSV and Excel buttons are used instead.
    """

    st.markdown(
        """
<style>
div[data-testid="stDataFrame"]
div[data-testid="stElementToolbar"] {
    display: none !important;
}

div[data-testid="stDataFrame"]
button[title="Download as CSV"] {
    display: none !important;
}

div[data-testid="stDataFrame"]
button[aria-label="Download as CSV"] {
    display: none !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _table_column_config(
    language: str,
) -> dict:
    """Return shared Streamlit table formatting."""

    current_spend = (
        "Mevcut Harcama"
        if language == "tr"
        else "Current Spend"
    )

    recommended_budget = (
        "Önerilen Bütçe"
        if language == "tr"
        else "Recommended Budget"
    )

    budget_change = (
        "Bütçe Değişimi %"
        if language == "tr"
        else "Budget Change %"
    )

    predicted_roas = (
        "Tahmini ROAS"
        if language == "tr"
        else "Predicted ROAS"
    )

    confidence = (
        "Öneri Güveni"
        if language == "tr"
        else "Recommendation Confidence"
    )

    opportunity_score = (
        "Fırsat Skoru"
        if language == "tr"
        else "Opportunity Score"
    )

    return {
        current_spend: st.column_config.NumberColumn(
            current_spend,
            format="₺%.2f",
        ),
        recommended_budget: (
            st.column_config.NumberColumn(
                recommended_budget,
                format="₺%.2f",
            )
        ),
        budget_change: (
            st.column_config.NumberColumn(
                budget_change,
                format="%.1f%%",
            )
        ),
        predicted_roas: (
            st.column_config.NumberColumn(
                predicted_roas,
                format="%.2fx",
            )
        ),
        confidence: (
            st.column_config.ProgressColumn(
                confidence,
                min_value=0.0,
                max_value=1.0,
                format="%.0f%%",
            )
        ),
        opportunity_score: (
            st.column_config.NumberColumn(
                opportunity_score,
                format="%.1f",
            )
        ),
    }


def render_optimization_table(
    enriched_recommendations: pd.DataFrame,
    language: str,
    limit: int = 10,
) -> None:
    """Render the principal optimization table."""

    hide_native_dataframe_toolbar()

    st.subheader(
        localized_text(
            language,
            "Optimizasyon Önerileri",
            "Optimization Recommendations",
        )
    )

    st.info(
        localized_text(
            language,
            (
                "Bu öneriler son pipeline çıktısını "
                "temsil eder. Öneri dosyasında analiz "
                "başlangıç ve bitiş tarihleri bulunmadığı "
                "için tarih filtresine dinamik olarak "
                "bağlanmamıştır."
            ),
            (
                "These recommendations represent the "
                "latest pipeline output. They are not "
                "dynamically linked to the date filter "
                "because the recommendation file does "
                "not contain analysis start and end dates."
            ),
        )
    )

    display_table = build_display_table(
        enriched_recommendations,
        language,
    )

    if display_table.empty:
        st.info(
            localized_text(
                language,
                "Optimizasyon önerisi bulunmuyor.",
                "No optimization recommendations are available.",
            )
        )
        return

    render_export_buttons(
        csv_dataframe=display_table,
        excel_sheets={
            "Optimization Recommendations": (
                display_table
            )
        },
        file_name="optimization_recommendations",
        language=language,
        key_prefix="optimization_table",
    )

    st.dataframe(
        display_table.head(limit),
        width="stretch",
        hide_index=True,
        column_config=_table_column_config(
            language
        ),
        key="executive_optimization_table",
    )


def _build_ranked_display(
    dataframe: pd.DataFrame,
    language: str,
    table_type: str,
) -> pd.DataFrame:
    """Build a concise opportunity or risk table."""

    if dataframe.empty:
        return pd.DataFrame()

    campaign_label = (
        "Kampanya"
        if language == "tr"
        else "Campaign"
    )

    campaign_type_label = (
        "Kampanya Türü"
        if language == "tr"
        else "Campaign Type"
    )

    current_spend_label = (
        "Mevcut Harcama"
        if language == "tr"
        else "Current Spend"
    )

    predicted_roas_label = (
        "Tahmini ROAS"
        if language == "tr"
        else "Predicted ROAS"
    )

    score_label = (
        (
            "Fırsat Skoru"
            if language == "tr"
            else "Opportunity Score"
        )
        if table_type == "opportunity"
        else (
            "Risk Skoru"
            if language == "tr"
            else "Risk Score"
        )
    )

    score_column = (
        "OpportunityScoreCanonical"
        if table_type == "opportunity"
        else "RiskScoreCanonical"
    )

    campaign_type_map = {
        "Brand": (
            "Marka"
            if language == "tr"
            else "Brand"
        ),
        "Generic": (
            "Genel"
            if language == "tr"
            else "Generic"
        ),
    }

    return pd.DataFrame(
        {
            campaign_label: dataframe[
                "CampaignCanonical"
            ],
            campaign_type_label: dataframe[
                "CampaignTypeCanonical"
            ].replace(
                campaign_type_map
            ),
            current_spend_label: dataframe[
                "CurrentSpendCanonical"
            ],
            predicted_roas_label: dataframe[
                "PredictedROASCanonical"
            ],
            score_label: dataframe[
                score_column
            ],
        }
    )


def render_opportunity_risk_tables(
    opportunities: pd.DataFrame,
    risks: pd.DataFrame,
    language: str,
) -> None:
    """Render concise top opportunities and risks."""

    hide_native_dataframe_toolbar()

    opportunity_column, risk_column = (
        st.columns(2)
    )

    with opportunity_column:
        st.markdown(
            localized_text(
                language,
                "**İlk 3 Fırsat**",
                "**Top 3 Opportunities**",
            )
        )

        opportunity_display = (
            _build_ranked_display(
                opportunities,
                language,
                "opportunity",
            )
        )

        if opportunity_display.empty:
            st.info(
                localized_text(
                    language,
                    "Fırsat verisi bulunmuyor.",
                    "Opportunity data is not available.",
                )
            )
        else:
            opportunity_score_label = (
                "Fırsat Skoru"
                if language == "tr"
                else "Opportunity Score"
            )

            st.dataframe(
                opportunity_display,
                width="stretch",
                hide_index=True,
                column_config={
                    **_table_column_config(
                        language
                    ),
                    opportunity_score_label: (
                        st.column_config.NumberColumn(
                            opportunity_score_label,
                            format="%.1f",
                        )
                    ),
                },
                key="executive_opportunity_table",
            )

    with risk_column:
        st.markdown(
            localized_text(
                language,
                "**İlk 3 Risk**",
                "**Top 3 Risks**",
            )
        )

        risk_display = _build_ranked_display(
            risks,
            language,
            "risk",
        )

        if risk_display.empty:
            st.info(
                localized_text(
                    language,
                    "Risk verisi bulunmuyor.",
                    "Risk data is not available.",
                )
            )
        else:
            risk_score_label = (
                "Risk Skoru"
                if language == "tr"
                else "Risk Score"
            )

            st.dataframe(
                risk_display,
                width="stretch",
                hide_index=True,
                column_config={
                    **_table_column_config(
                        language
                    ),
                    risk_score_label: (
                        st.column_config.NumberColumn(
                            risk_score_label,
                            format="%.1f",
                        )
                    ),
                },
                key="executive_risk_table",
            )


def render_data_tab(
    dataframe: pd.DataFrame,
    sheet_name: str,
    file_name: str,
    key_prefix: str,
    empty_message_tr: str,
    empty_message_en: str,
    language: str,
) -> None:
    """
    Render one data tab with explicit CSV and Excel buttons.
    """

    if dataframe.empty:
        st.info(
            localized_text(
                language,
                empty_message_tr,
                empty_message_en,
            )
        )
        return

    render_export_buttons(
        csv_dataframe=dataframe,
        excel_sheets={
            sheet_name: dataframe,
        },
        file_name=file_name,
        language=language,
        key_prefix=key_prefix,
    )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        key=f"{key_prefix}_dataframe",
    )


def render_detailed_data(
    recommendation_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    daily: pd.DataFrame,
    language: str,
) -> None:
    """
    Render detailed data tabs with explicit
    CSV and Excel downloads.
    """

    hide_native_dataframe_toolbar()

    with st.expander(
        localized_text(
            language,
            "Detaylı Veri Görünümü",
            "Detailed Data View",
        ),
        expanded=False,
    ):
        summary_tab, portfolio_tab, daily_tab = (
            st.tabs(
                [
                    localized_text(
                        language,
                        "Öneri Özeti",
                        "Recommendation Summary",
                    ),
                    localized_text(
                        language,
                        "Portföy Verisi",
                        "Portfolio Data",
                    ),
                    localized_text(
                        language,
                        "Günlük Veri",
                        "Daily Data",
                    ),
                ]
            )
        )

        with summary_tab:
            render_data_tab(
                dataframe=recommendation_summary,
                sheet_name="Recommendation Summary",
                file_name="recommendation_summary",
                key_prefix=(
                    "detailed_recommendation_summary"
                ),
                empty_message_tr=(
                    "Öneri özeti bulunmuyor."
                ),
                empty_message_en=(
                    "Recommendation summary is not available."
                ),
                language=language,
            )

        with portfolio_tab:
            render_data_tab(
                dataframe=portfolio,
                sheet_name="Portfolio",
                file_name="portfolio_data",
                key_prefix="detailed_portfolio",
                empty_message_tr=(
                    "Portföy verisi bulunmuyor."
                ),
                empty_message_en=(
                    "Portfolio data is not available."
                ),
                language=language,
            )

        with daily_tab:
            render_data_tab(
                dataframe=daily,
                sheet_name="Daily Data",
                file_name="daily_advertising_data",
                key_prefix="detailed_daily",
                empty_message_tr=(
                    "Günlük veri bulunmuyor."
                ),
                empty_message_en=(
                    "Daily data is not available."
                ),
                language=language,
            )