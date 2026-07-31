from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from dashboard_demo.layout import localized_text
from dashboard_demo.services.executive_data import (
    DateCoverage,
)
from dashboard_demo.services.executive_metrics import (
    ExecutiveKPIs,
    KPIComparison,
)
from dashboard_demo.services.executive_scoring import (
    ActionSummary,
)


def _localized_decimal(
    value: float,
    decimals: int,
    language: str,
) -> str:
    """Format decimal and thousands separators."""

    formatted = f"{value:,.{decimals}f}"

    if language != "tr":
        return formatted

    return (
        formatted
        .replace(",", "__THOUSANDS__")
        .replace(".", ",")
        .replace("__THOUSANDS__", ".")
    )


def compact_currency(
    value: float,
    language: str,
) -> str:
    """Format currency for executive cards."""

    absolute_value = abs(value)

    if absolute_value >= 1_000_000:
        suffix = (
            "Mn"
            if language == "tr"
            else "M"
        )

        compact_value = _localized_decimal(
            value / 1_000_000,
            2,
            language,
        )

        return f"₺{compact_value} {suffix}"

    if absolute_value >= 1_000:
        suffix = (
            "B"
            if language == "tr"
            else "K"
        )

        compact_value = _localized_decimal(
            value / 1_000,
            1,
            language,
        )

        return f"₺{compact_value} {suffix}"

    return (
        "₺"
        + _localized_decimal(
            value,
            0,
            language,
        )
    )


def format_number(
    value: float,
    language: str,
    decimals: int = 0,
) -> str:
    """Format a localized number."""

    return _localized_decimal(
        value,
        decimals,
        language,
    )


def format_ratio(
    value: float,
    language: str,
) -> str:
    """Format a ratio such as ROAS."""

    return (
        _localized_decimal(
            value,
            2,
            language,
        )
        + "x"
    )


def format_percentage(
    value: float,
    language: str,
    decimals: int = 1,
) -> str:
    """Format a localized percentage."""

    return (
        "%"
        + _localized_decimal(
            value,
            decimals,
            language,
        )
    )


def format_delta(
    value: Optional[float],
) -> Optional[str]:
    """Format a Streamlit metric delta."""

    if value is None:
        return None

    return f"{value:+.1f}%"


def render_data_coverage(
    current_coverage: DateCoverage,
    comparison_coverage: DateCoverage,
    comparison_enabled: bool,
    data_age_days: Optional[int],
    language: str,
) -> None:
    """Render current and comparison data coverage."""

    first, second, third = st.columns(3)

    first.metric(
        localized_text(
            language,
            "Seçilen Gün",
            "Selected Days",
        ),
        current_coverage.selected_days,
    )

    second.metric(
        localized_text(
            language,
            "Veri Bulunan Gün",
            "Days With Data",
        ),
        current_coverage.available_days,
    )

    third.metric(
        localized_text(
            language,
            "Veri Kapsama Oranı",
            "Data Coverage",
        ),
        format_percentage(
            current_coverage.coverage_ratio * 100,
            language,
        ),
    )

    st.progress(
        current_coverage.coverage_ratio
    )

    if (
        current_coverage.available_start
        and current_coverage.available_end
    ):
        st.caption(
            localized_text(
                language,
                (
                    "Mevcut veri dönemi: "
                    f"{current_coverage.available_start:%d.%m.%Y}"
                    " — "
                    f"{current_coverage.available_end:%d.%m.%Y}"
                ),
                (
                    "Available data period: "
                    f"{current_coverage.available_start:%d.%m.%Y}"
                    " — "
                    f"{current_coverage.available_end:%d.%m.%Y}"
                ),
            )
        )

    if (
        data_age_days is not None
        and data_age_days > 1
    ):
        st.warning(
            localized_text(
                language,
                (
                    "Veri güncel değil: son kayıt "
                    f"{data_age_days} gün önce oluşturuldu."
                ),
                (
                    "Data is not current: the latest "
                    f"record is {data_age_days} days old."
                ),
            )
        )

    if comparison_enabled:
        comparison_percentage = (
            comparison_coverage.coverage_ratio
            * 100
        )

        if not comparison_coverage.is_sufficient():
            st.warning(
                localized_text(
                    language,
                    (
                        "Karşılaştırma dönemi için veri "
                        "kapsamı yetersiz: "
                        f"{comparison_coverage.available_days}/"
                        f"{comparison_coverage.selected_days} gün "
                        f"(%{comparison_percentage:.1f}). "
                        "KPI değişim yüzdeleri gösterilmeyecek."
                    ),
                    (
                        "Comparison-period coverage is "
                        "insufficient: "
                        f"{comparison_coverage.available_days}/"
                        f"{comparison_coverage.selected_days} days "
                        f"({comparison_percentage:.1f}%). "
                        "KPI change percentages will not be shown."
                    ),
                )
            )
        else:
            st.info(
                localized_text(
                    language,
                    (
                        "Karşılaştırma kapsamı: "
                        f"%{comparison_percentage:.1f}"
                    ),
                    (
                        "Comparison coverage: "
                        f"{comparison_percentage:.1f}%"
                    ),
                )
            )


def render_kpi_cards(
    current: ExecutiveKPIs,
    comparison: KPIComparison,
    target_roas: float,
    language: str,
) -> None:
    """Render the six core executive KPI cards."""

    st.subheader(
        localized_text(
            language,
            "Temel Performans Göstergeleri",
            "Core Performance Indicators",
        )
    )

    first_row = st.columns(3)
    second_row = st.columns(3)

    first_row[0].metric(
        localized_text(
            language,
            "Gelir",
            "Revenue",
        ),
        compact_currency(
            current.revenue,
            language,
        ),
        delta=format_delta(
            comparison.revenue_change_pct
        ),
    )

    first_row[1].metric(
        localized_text(
            language,
            "Reklam Harcaması",
            "Ad Spend",
        ),
        compact_currency(
            current.spend,
            language,
        ),
        delta=format_delta(
            comparison.spend_change_pct
        ),
        delta_color="inverse",
    )

    first_row[2].metric(
        "ROAS",
        format_ratio(
            current.roas,
            language,
        ),
        delta=format_delta(
            comparison.roas_change_pct
        ),
    )

    second_row[0].metric(
        localized_text(
            language,
            "Dönüşümler",
            "Conversions",
        ),
        format_number(
            current.conversions,
            language,
        ),
        delta=format_delta(
            comparison.conversions_change_pct
        ),
    )

    second_row[1].metric(
        localized_text(
            language,
            "Dönüşüm Başına Maliyet",
            "Cost per Acquisition",
        ),
        compact_currency(
            current.cpa,
            language,
        ),
        delta=format_delta(
            comparison.cpa_change_pct
        ),
        delta_color="inverse",
    )

    target_gap_percentage = (
        (
            current.roas - target_roas
        )
        / target_roas
        * 100
        if target_roas > 0
        else 0.0
    )

    target_delta = localized_text(
        language,
        (
            "Gerçekleşen "
            f"{format_ratio(current.roas, language)}"
            f" · {target_gap_percentage:+.1f}%"
        ),
        (
            "Actual "
            f"{format_ratio(current.roas, language)}"
            f" · {target_gap_percentage:+.1f}%"
        ),
    )

    second_row[2].metric(
        localized_text(
            language,
            "Hedef ROAS",
            "Target ROAS",
        ),
        format_ratio(
            target_roas,
            language,
        ),
        delta=target_delta,
    )


def render_action_cards(
    summary: ActionSummary,
    model_r2: Optional[float],
    language: str,
) -> None:
    """Render the Executive Action Center cards."""

    st.subheader(
        localized_text(
            language,
            "Yönetici Aksiyon Merkezi",
            "Executive Action Center",
        )
    )

    first_row = st.columns(4)

    first_row[0].metric(
        localized_text(
            language,
            "Bütçe Artır",
            "Increase Budget",
        ),
        summary.increase_count,
    )

    first_row[1].metric(
        localized_text(
            language,
            "Bütçe Azalt",
            "Reduce Budget",
        ),
        summary.reduce_count,
    )

    first_row[2].metric(
        localized_text(
            language,
            "Bütçeyi Koru",
            "Maintain Budget",
        ),
        summary.maintain_count,
    )

    first_row[3].metric(
        localized_text(
            language,
            "İncele",
            "Review",
        ),
        summary.review_count,
    )

    second_row = st.columns(4)

    second_row[0].metric(
        localized_text(
            language,
            "Yüksek Risk",
            "High Risk",
        ),
        summary.high_risk_count,
    )

    second_row[1].metric(
        localized_text(
            language,
            "Model Doğrulama R²",
            "Model Validation R²",
        ),
        (
            format_percentage(
                model_r2,
                language,
            )
            if model_r2 is not None
            else localized_text(
                language,
                "Veri Yok",
                "No Data",
            )
        ),
    )

    confidence_value = (
        localized_text(
            language,
            "Yüksek",
            "High",
        )
        if (
            summary.high_confidence_count
            == summary.total_recommendation_count
            and summary.total_recommendation_count > 0
        )
        else localized_text(
            language,
            "Karma",
            "Mixed",
        )
    )

    confidence_detail = (
        f"{summary.high_confidence_count}/"
        f"{summary.total_recommendation_count} "
        + localized_text(
            language,
            "kampanya",
            "campaigns",
        )
    )

    second_row[2].metric(
        localized_text(
            language,
            "Öneri Güveni",
            "Recommendation Confidence",
        ),
        confidence_value,
        delta=confidence_detail,
        delta_color="off",
    )

    second_row[3].metric(
        localized_text(
            language,
            "Veri Yetersiz",
            "Insufficient Data",
        ),
        summary.insufficient_data_count,
    )


def render_ai_executive_summary(
    current_kpis: ExecutiveKPIs,
    action_summary: ActionSummary,
    opportunities: pd.DataFrame,
    risks: pd.DataFrame,
    target_roas: float,
    executive_commentary: str,
    language: str,
) -> None:
    """Render a structured executive AI summary."""

    st.subheader(
        localized_text(
            language,
            "Yönetici AI Özeti",
            "Executive AI Summary",
        )
    )

    if current_kpis.roas >= target_roas:
        general_status = localized_text(
            language,
            (
                "Genel durum güçlü. "
                f"ROAS {current_kpis.roas:.2f}x ve "
                f"{target_roas:.2f}x hedefinin üzerinde."
            ),
            (
                "Overall performance is strong. "
                f"ROAS is {current_kpis.roas:.2f}x, "
                f"above the {target_roas:.2f}x target."
            ),
        )
    else:
        general_status = localized_text(
            language,
            (
                "Genel performans dikkat gerektiriyor. "
                f"ROAS {current_kpis.roas:.2f}x ve "
                f"{target_roas:.2f}x hedefinin altında."
            ),
            (
                "Overall performance requires attention. "
                f"ROAS is {current_kpis.roas:.2f}x, "
                f"below the {target_roas:.2f}x target."
            ),
        )

    opportunity_campaign = localized_text(
        language,
        "Belirlenemedi",
        "Not available",
    )

    if (
        not opportunities.empty
        and "CampaignCanonical"
        in opportunities.columns
    ):
        opportunity_campaign = str(
            opportunities.iloc[0][
                "CampaignCanonical"
            ]
        )

    risk_campaign = localized_text(
        language,
        "Belirlenemedi",
        "Not available",
    )

    if (
        not risks.empty
        and "CampaignCanonical"
        in risks.columns
    ):
        risk_campaign = str(
            risks.iloc[0][
                "CampaignCanonical"
            ]
        )

    opportunity_text = localized_text(
        language,
        (
            "En önemli fırsat: "
            f"{opportunity_campaign}. "
            "Artış sinyali bulunan kampanya sayısı: "
            f"{action_summary.increase_count}."
        ),
        (
            "Top opportunity: "
            f"{opportunity_campaign}. "
            "Campaigns with an increase signal: "
            f"{action_summary.increase_count}."
        ),
    )

    risk_text = localized_text(
        language,
        (
            "En önemli risk: "
            f"{risk_campaign}. "
            "Azaltma sinyali bulunan kampanya sayısı: "
            f"{action_summary.reduce_count}."
        ),
        (
            "Top risk: "
            f"{risk_campaign}. "
            "Campaigns with a reduction signal: "
            f"{action_summary.reduce_count}."
        ),
    )

    status_column, opportunity_column, risk_column = (
        st.columns(3)
    )

    with status_column:
        st.info(
            general_status
        )

    with opportunity_column:
        st.success(
            opportunity_text
        )

    with risk_column:
        st.warning(
            risk_text
        )

    if language == "tr":
        detailed_commentary = (
            "Yüksek güvenli ve hedefin üzerinde ROAS "
            "üreten kampanyalara kontrollü bütçe "
            "aktarılması; hedefin altında kalan veya "
            "bütçe artış riski taşıyan kampanyaların "
            "yakından izlenmesi önerilir. "
            f"{action_summary.insufficient_data_count} "
            "kampanyada aktif veri yetersiz olduğu için "
            "manuel inceleme yapılmalıdır."
        )
    else:
        detailed_commentary = (
            executive_commentary.strip()
            if executive_commentary.strip()
            else (
                "Consider reallocating budget toward "
                "high-confidence campaigns that generate "
                "ROAS above target. Closely monitor "
                "campaigns below target or carrying "
                "budget-increase risk. "
                f"{action_summary.insufficient_data_count} "
                "campaigns require manual review because "
                "active data is insufficient."
            )
        )

    with st.expander(
        localized_text(
            language,
            "Detaylı AI Yorumu",
            "Detailed AI Commentary",
        ),
        expanded=False,
    ):
        st.write(
            detailed_commentary
        )


