from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import streamlit as st

from dashboard_demo.filters import (
    COMPARISON_KEYS,
    DATE_PRESET_KEYS,
    DashboardFilters,
    get_comparison_range,
    resolve_date_range,
)
from dashboard_demo.i18n import translate
from dashboard_demo.services.analysis_runner_demo import (
    get_source_status,
    run_analysis_for_period,
)


@dataclass(frozen=True)
class DashboardContext:
    """Bütün dashboard sayfalarında kullanılan ortak bağlam."""

    filters: DashboardFilters
    language: str


def inject_global_styles() -> None:
    """Ortak dashboard görünümünü yükle."""

    st.markdown(
        """
<style>
.block-container {
    max-width: 1450px;
    padding-top: 4.5rem !important;
    padding-bottom: 5rem;
}

header[data-testid="stHeader"] {
    height: 3.5rem;
}

div[data-testid="stAppViewContainer"] {
    overflow: visible;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

.platform-brand {
    padding: 0.35rem 0 0.85rem 0;
}

.platform-brand-title {
    font-size: 1.15rem;
    font-weight: 850;
    line-height: 1.25;
}

.platform-brand-subtitle {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 0.35rem;
}

.page-hero {
    padding: 1.55rem 1.75rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 20px;
    background:
        radial-gradient(
            circle at top right,
            rgba(59, 130, 246, 0.16),
            transparent 35%
        ),
        rgba(15, 23, 42, 0.78);
    margin-bottom: 1.2rem;
}

.page-eyebrow {
    color: #7dd3fc;
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}

.page-title {
    font-size: clamp(1.9rem, 3.2vw, 2.7rem);
    font-weight: 850;
    line-height: 1.12;
    margin: 0;
}

.page-subtitle {
    color: #aab4c3;
    line-height: 1.6;
    margin-top: 0.65rem;
    margin-bottom: 0;
    max-width: 920px;
}

.filter-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 1rem;
}

.filter-chip {
    padding: 0.38rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.48);
    color: #cbd5e1;
    font-size: 0.8rem;
}

div[data-testid="stButton"] > button {
    border-radius: 12px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 15px;
    padding: 0.8rem;
    background: rgba(15, 23, 42, 0.34);
}

.read-only-footer {
    color: #94a3b8;
    font-size: 0.8rem;
    line-height: 1.5;
    margin-top: 1rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand(language: str) -> None:
    """Sol menüde platform bilgisini göster."""

    html = (
        '<div class="platform-brand">'
        '<div class="platform-brand-title">'
        f'{translate("app_name", language)}'
        "</div>"
        '<div class="platform-brand-subtitle">'
        f'{translate("platform_subtitle", language)}'
        "</div>"
        "</div>"
    )

    st.sidebar.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "Advertising Decision Intelligence",
) -> None:
    """Ortak sayfa başlığını göster."""

    html = (
        '<div class="page-hero">'
        '<div class="page-eyebrow">'
        f"{eyebrow}"
        "</div>"
        '<h1 class="page-title">'
        f"{title}"
        "</h1>"
        '<p class="page-subtitle">'
        f"{subtitle}"
        "</p>"
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def render_interactive_filter_bar(
    default_preset: str = "last_30_days",
    default_comparison: str = "previous_period",
    reference_date: date | None = None,
) -> DashboardFilters:
    """
    Ortak dil, analiz dönemi ve karşılaştırma kontrollerini göster.

    API bağlantıları hazırsa göreli tarihler dünden geriye
    hesaplanır. API bağlantıları hazır değilse mevcut CSV
    verisinin son tarihi kullanılır.
    """

    if "dashboard_language" not in st.session_state:
        st.session_state["dashboard_language"] = "tr"

    source_status = get_source_status()

    sources_ready = bool(
        source_status["google_ads"]
        and source_status["ga4"]
    )

    date_reference = (
        date.today() - timedelta(days=1)
        if sources_ready
        else reference_date
    )

    language_column, date_column = st.columns([1, 3])

    # ---------------------------------------------------------
    # DİL
    # ---------------------------------------------------------
    with language_column:
        language = st.selectbox(
            "Dil / Language",
            options=["tr", "en"],
            format_func=lambda value: (
                "Dil: Türkçe"
                if value == "tr"
                else "Language: English"
            ),
            key="dashboard_language",
            label_visibility="collapsed",
        )

    # ---------------------------------------------------------
    # ANALİZ DÖNEMİ OTURUM DURUMU
    # ---------------------------------------------------------
    preset_state_key = (
        f"dashboard_toolbar_date_preset_{language}"
    )

    stored_preset = st.session_state.get(
        "dashboard_selected_preset",
        default_preset,
    )

    if stored_preset not in DATE_PRESET_KEYS:
        stored_preset = default_preset

    if (
        preset_state_key not in st.session_state
        or st.session_state[preset_state_key]
        not in DATE_PRESET_KEYS
    ):
        st.session_state[preset_state_key] = stored_preset

    # ---------------------------------------------------------
    # KARŞILAŞTIRMA DÖNEMİ OTURUM DURUMU
    # ---------------------------------------------------------
    comparison_state_key = (
        f"dashboard_toolbar_comparison_{language}"
    )

    stored_comparison = st.session_state.get(
        "dashboard_selected_comparison",
        default_comparison,
    )

    if stored_comparison not in COMPARISON_KEYS:
        stored_comparison = default_comparison

    if (
        comparison_state_key not in st.session_state
        or st.session_state[comparison_state_key]
        not in COMPARISON_KEYS
    ):
        st.session_state[
            comparison_state_key
        ] = stored_comparison

    comparison_default_labels = {
        "custom_comparison": (
            "Özel Karşılaştırma Tarihleri"
            if language == "tr"
            else "Custom Comparison Dates"
        ),
    }

    custom_comparison_start = None
    custom_comparison_end = None

    # ---------------------------------------------------------
    # TARİH
    # ---------------------------------------------------------
    with date_column:
        date_title = (
            "Tarih"
            if language == "tr"
            else "Date"
        )

        with st.expander(
            date_title,
            expanded=False,
        ):
            preset = st.selectbox(
                (
                    "Analiz Dönemi"
                    if language == "tr"
                    else "Analysis Period"
                ),
                options=DATE_PRESET_KEYS,
                format_func=lambda value: translate(
                    value,
                    language,
                ),
                key=preset_state_key,
            )

            comparison = st.selectbox(
                (
                    "Karşılaştırma Dönemi"
                    if language == "tr"
                    else "Comparison Period"
                ),
                options=COMPARISON_KEYS,
                format_func=lambda value: translate(
                    value,
                    language,
                    default=comparison_default_labels.get(
                        value,
                        value,
                    ),
                ),
                key=comparison_state_key,
            )

            start_date, end_date = resolve_date_range(
                preset,
                today=date_reference,
            )

            # Özel analiz dönemi
            if preset == "custom_range":
                selected_dates = st.date_input(
                    (
                        "Özel Analiz Tarihleri"
                        if language == "tr"
                        else "Custom Analysis Dates"
                    ),
                    value=(start_date, end_date),
                    key=(
                        "dashboard_toolbar_"
                        "custom_date_range"
                    ),
                )

                if (
                    isinstance(selected_dates, tuple)
                    and len(selected_dates) == 2
                ):
                    start_date, end_date = selected_dates

            start_date, end_date = (
                min(start_date, end_date),
                max(start_date, end_date),
            )

            # Özel karşılaştırma dönemi
            if comparison == "custom_comparison":
                default_comparison_range = (
                    get_comparison_range(
                        start_date,
                        end_date,
                        "previous_period",
                    )
                )

                assert default_comparison_range is not None

                selected_comparison_dates = st.date_input(
                    (
                        "Özel Karşılaştırma Tarihleri"
                        if language == "tr"
                        else "Custom Comparison Dates"
                    ),
                    value=default_comparison_range,
                    key=(
                        "dashboard_toolbar_"
                        "custom_comparison_range"
                    ),
                )

                if (
                    isinstance(
                        selected_comparison_dates,
                        tuple,
                    )
                    and len(
                        selected_comparison_dates
                    ) == 2
                ):
                    (
                        custom_comparison_start,
                        custom_comparison_end,
                    ) = selected_comparison_dates

            st.divider()

            run_button_label = (
                "Seçilen Dönemi Analiz Et"
                if language == "tr"
                else "Analyze Selected Period"
            )

            if st.button(
                run_button_label,
                type="primary",
                width="stretch",
                disabled=not sources_ready,
                key="dashboard_run_selected_period",
            ):
                # Widget anahtarlarını değiştirmeden seçimi sakla.
                st.session_state[
                    "dashboard_selected_preset"
                ] = preset

                st.session_state[
                    "dashboard_selected_comparison"
                ] = comparison

                spinner_text = (
                    "Google Ads ve GA4 verileri çekiliyor; "
                    "model ve öneriler yeniden oluşturuluyor..."
                    if language == "tr"
                    else
                    "Loading Google Ads and GA4 data; rebuilding "
                    "models and recommendations..."
                )

                with st.spinner(spinner_text):
                    run_result = run_analysis_for_period(
                        start_date,
                        end_date,
                    )

                if run_result.success:
                    st.session_state[
                        "dashboard_analysis_message"
                    ] = (
                        "success",
                        run_result.message,
                    )

                    st.cache_data.clear()
                    st.rerun()

                else:
                    st.session_state[
                        "dashboard_analysis_message"
                    ] = (
                        "error",
                        run_result.message,
                    )

                    if run_result.log_tail:
                        with st.expander(
                            (
                                "Pipeline çalışma kaydı"
                                if language == "tr"
                                else "Pipeline run log"
                            )
                        ):
                            st.code(run_result.log_tail)

            if not sources_ready:
                missing_sources = []

                if not source_status["google_ads"]:
                    missing_sources.append("Google Ads")

                if not source_status["ga4"]:
                    missing_sources.append("GA4")

                st.caption(
                    (
                        "Analiz başlatılamıyor. "
                        "Eksik API yapılandırması: "
                        if language == "tr"
                        else
                        "Analysis cannot start. "
                        "Missing API configuration: "
                    )
                    + ", ".join(missing_sources)
                )

            analysis_message = st.session_state.pop(
                "dashboard_analysis_message",
                None,
            )

            if analysis_message:
                message_type, message_text = analysis_message

                if message_type == "success":
                    st.success(message_text)
                else:
                    st.error(message_text)

    # Aktif seçimleri bütün sayfalarda kullanılmak üzere sakla.
    st.session_state[
        "dashboard_selected_preset"
    ] = preset

    st.session_state[
        "dashboard_selected_comparison"
    ] = comparison

    comparison_range = get_comparison_range(
        start_date,
        end_date,
        comparison,
        custom_start_date=custom_comparison_start,
        custom_end_date=custom_comparison_end,
    )

    st.caption(
        (
            "Dil ve tarih seçimleri tüm analizlere uygulanır."
            if language == "tr"
            else (
                "Language and date selections "
                "apply to all analyses."
            )
        )
    )

    return DashboardFilters(
        language=language,
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        comparison=comparison,
        comparison_start_date=(
            comparison_range[0]
            if comparison_range
            else None
        ),
        comparison_end_date=(
            comparison_range[1]
            if comparison_range
            else None
        ),
    )


def render_filter_summary(
    filters: DashboardFilters,
) -> None:
    """Aktif tarih, karşılaştırma ve dil seçimini göster."""

    comparison_label = translate(
        filters.comparison,
        filters.language,
        default=(
            "Özel Karşılaştırma Tarihleri"
            if filters.language == "tr"
            else "Custom Comparison Dates"
        ),
    )

    comparison_dates = ""

    if (
        filters.comparison_start_date
        and filters.comparison_end_date
    ):
        comparison_dates = (
            " · "
            f"{filters.comparison_start_date:%d.%m.%Y}"
            " — "
            f"{filters.comparison_end_date:%d.%m.%Y}"
        )

    language_label = (
        "Türkçe"
        if filters.language == "tr"
        else "English"
    )

    html = (
        '<div class="filter-summary">'
        '<div class="filter-chip">'
        f"📅 {filters.start_date:%d.%m.%Y}"
        f" — {filters.end_date:%d.%m.%Y}"
        "&nbsp;·&nbsp;"
        f"↔ {comparison_label}{comparison_dates}"
        "</div>"
        '<div class="filter-chip">'
        f"🌐 {language_label}"
        "</div>"
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def initialize_dashboard(
    page_title: str,
    page_icon: str,
    title: str,
    subtitle: str,
    eyebrow: str = "Advertising Decision Intelligence",
    default_preset: str = "last_30_days",
    default_comparison: str = "previous_period",
    reference_date: date | None = None,
) -> DashboardContext:
    """
    Bütün dashboard sayfalarının ortak ayarlarını hazırla.

    reference_date parametresi app.py ve bütün sayfa
    dosyalarıyla uyumludur.
    """

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
    )

    inject_global_styles()

    filters = render_interactive_filter_bar(
        default_preset=default_preset,
        default_comparison=default_comparison,
        reference_date=reference_date,
    )

    render_sidebar_brand(filters.language)

    render_page_header(
        title=title,
        subtitle=subtitle,
        eyebrow=eyebrow,
    )

    render_filter_summary(filters)

    return DashboardContext(
        filters=filters,
        language=filters.language,
    )


def render_read_only_footer(
    language: str,
) -> None:
    """Ortak bilgilendirme notunu göster."""

    html = (
        '<div class="read-only-footer">'
        f'{translate("read_only", language)}'
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def localized_text(
    language: str,
    turkish: str,
    english: str,
) -> str:
    """Aktif dile göre metni döndür."""

    return (
        turkish
        if language == "tr"
        else english
    )


def format_status_value(
    value: Any,
    language: str,
) -> str:
    """Ortak sistem durum değerlerini biçimlendir."""

    if isinstance(value, bool):
        return (
            translate("online", language)
            if value
            else translate("offline", language)
        )

    return str(value)




