from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
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


_DATE_COLUMN_CANDIDATES = {
    "date",
    "day",
    "report_date",
    "reportdate",
    "segments_date",
    "segmentsdate",
}


_DEMO_FILTER_STATE_VERSION = 2


def _parse_demo_date(value: Any) -> date | None:
    """CSV içindeki olası tarih değerini güvenli biçimde çöz."""

    raw_value = str(value).strip()

    if not raw_value:
        return None

    iso_candidate = raw_value[:10]

    try:
        return date.fromisoformat(iso_candidate)
    except ValueError:
        pass

    for date_format in (
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(
                raw_value,
                date_format,
            ).date()
        except ValueError:
            continue

    return None


def _get_demo_reference_date() -> date:
    """
    Public demo için kullanılacak sanal bugünü belirle.

    Tarih yalnızca proje kökündeki anonim ``demo_data`` klasöründe
    bulunan ``ads_daily_fact.csv`` dosyasından okunur. Canlı çıktı
    klasörleri ve canlı API kaynakları referans alınmaz.
    """

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "demo_data" / "ads_daily_fact.csv"

    latest_date: date | None = None

    if csv_path.is_file():
        try:
            with csv_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as csv_file:
                reader = csv.DictReader(csv_file)

                if reader.fieldnames:
                    date_column = next(
                        (
                            column
                            for column in reader.fieldnames
                            if column.strip()
                            .lower()
                            .replace("-", "_")
                            .replace(" ", "_")
                            in _DATE_COLUMN_CANDIDATES
                        ),
                        None,
                    )

                    if date_column is not None:
                        for row in reader:
                            parsed_date = _parse_demo_date(
                                row.get(date_column)
                            )

                            if (
                                parsed_date is not None
                                and (
                                    latest_date is None
                                    or parsed_date > latest_date
                                )
                            ):
                                latest_date = parsed_date

        except (
            OSError,
            UnicodeError,
            csv.Error,
        ):
            latest_date = None

    if latest_date is not None:
        return latest_date

    # Demo CSV bulunamazsa uygulamanın açılmasını engelleme.
    return date.today() - timedelta(days=1)


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
    default_preset: str = "this_month",
    default_comparison: str = "no_comparison",
    reference_date: date | None = None,
) -> DashboardFilters:
    """
    Ortak dil, analiz dönemi ve karşılaştırma kontrollerini göster.

    Public demo sürümünde tarih aralıkları, anonim demo verisinin
    mevcut son tarihine göre hesaplanır. Canlı API çağrısı yapılmaz.
    """

    if "dashboard_language" not in st.session_state:
        st.session_state["dashboard_language"] = "tr"

    # Daha önce tarayıcı oturumunda saklanan eski "Son 30 Gün" ve
    # "Önceki Dönem" seçimlerini yalnızca bu sürüm ilk çalıştığında temizle.
    # Sonraki etkileşimlerde kullanıcının yeni seçimi korunur.
    if (
        st.session_state.get("dashboard_demo_filter_state_version")
        != _DEMO_FILTER_STATE_VERSION
    ):
        keys_to_reset = (
            "dashboard_selected_preset",
            "dashboard_selected_comparison",
            "dashboard_toolbar_date_preset_tr",
            "dashboard_toolbar_date_preset_en",
            "dashboard_toolbar_comparison_tr",
            "dashboard_toolbar_comparison_en",
            "dashboard_toolbar_custom_date_range",
            "dashboard_toolbar_custom_comparison_range",
        )

        for state_key in keys_to_reset:
            st.session_state.pop(state_key, None)

        st.session_state["dashboard_selected_preset"] = "this_month"
        st.session_state["dashboard_selected_comparison"] = (
            "no_comparison"
        )
        st.session_state["dashboard_demo_filter_state_version"] = (
            _DEMO_FILTER_STATE_VERSION
        )

    source_status = get_source_status()

    sources_ready = bool(
        source_status["google_ads"]
        and source_status["ga4"]
    )

    # Demo tarih filtreleri canlı güne göre değil, anonim veri setinin
    # mevcut son tarihine göre hesaplanır.
    date_reference = (
        reference_date
        if reference_date is not None
        else _get_demo_reference_date()
    )

    # Public demo bütün sayfalarda veri bulunan ayla açılır.
    # Eski sayfalardan "last_30_days" gönderilse bile demo başlangıcı
    # güvenli biçimde "this_month / no_comparison" olarak normalize edilir.
    default_preset = "this_month"
    default_comparison = "no_comparison"

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
                "Seçilen Dönemi Göster"
                if language == "tr"
                else "Show Selected Period"
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
                    "Seçilen dönem anonim demo verilerine uygulanıyor..."
                    if language == "tr"
                    else
                    "Applying the selected period to anonymized demo data..."
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
    default_preset: str = "this_month",
    default_comparison: str = "no_comparison",
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
    """Public demo bilgilendirme notunu göster."""

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




