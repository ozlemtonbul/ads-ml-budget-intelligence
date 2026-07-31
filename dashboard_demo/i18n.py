from __future__ import annotations

from typing import Final


DEFAULT_LANGUAGE: Final[str] = "tr"

TRANSLATIONS: dict[str, dict[str, str]] = {
    "tr": {
        "app_name": "Ads Budget Intelligence",
        "platform_subtitle": (
            "Kurumsal Reklam Karar Destek Platformu"
        ),
        "language": "Dil",
        "turkish": "Türkçe",
        "english": "English",
        "filters": "Filtreler",
        "date_range": "Tarih Aralığı",
        "comparison": "Karşılaştırma",
        "apply_filters": "Filtreleri Uygula",
        "reset_filters": "Filtreleri Sıfırla",

        "today": "Bugün",
        "yesterday": "Dün",
        "this_week": "Bu Hafta",
        "last_week": "Geçen Hafta",
        "last_7_days": "Son 7 Gün",
        "last_30_days": "Son 30 Gün",
        "last_60_days": "Son 60 Gün",
        "last_90_days": "Son 90 Gün",
        "this_month": "Bu Ay",
        "last_month": "Geçen Ay",
        "this_quarter": "Bu Çeyrek",
        "last_quarter": "Geçen Çeyrek",
        "this_year": "Yılbaşından Bugüne",
        "last_year": "Geçen Yıl",
        "custom_range": "Özel Tarih Aralığı",

        "no_comparison": "Karşılaştırma Yok",
        "previous_period": "Önceki Dönem",
        "previous_month": "Geçen Ay",
        "previous_quarter": "Geçen Çeyrek",
        "previous_year": "Geçen Yılın Aynı Dönemi",
        "previous_year_ytd": (
            "Geçen Yılın Aynı Yılbaşından Bugüne Dönemi"
        ),
        "custom_comparison": (
            "Özel Karşılaştırma Tarihleri"
        ),

        "comparison_length_warning": (
            "Analiz ve karşılaştırma dönemlerinin gün "
            "sayıları farklıdır. Yüzdesel değişimleri "
            "yorumlarken dönem uzunluğu farkını dikkate alın."
        ),

        "start_date": "Başlangıç Tarihi",
        "end_date": "Bitiş Tarihi",

        "filter_help": (
            "Analiz dönemini, karşılaştırma dönemini ve "
            "arayüz dilini seçin."
        ),

        "selection_applies_to_all": (
            "Seçimler tüm KPI, grafik, tablo ve yapay zekâ "
            "analizlerine uygulanır."
        ),

        "system_mode": "Sistem Modu",
        "online": "Çevrimiçi",
        "offline": "Çevrimdışı",
        "deterministic_mode": "Deterministik Mod",
        "hybrid_mode": "Hibrit Yapay Zekâ Modu",

        "executive_overview": "Yönetici Özeti",
        "campaign_analysis": "Kampanya Analizi",
        "budget_optimizer": "Bütçe Optimizasyonu",
        "ai_insights": "Yapay Zekâ Analizleri",
        "ask_ai": "Yapay Zekâ Asistanı",
        "system_status": "Sistem Durumu",
        "agency_recommendations": "Ajans Önerileri",
        "reports": "Raporlar",

        "last_update": "Son Güncelleme",
        "revenue": "Gelir",
        "ad_spend": "Reklam Harcaması",
        "roas": "ROAS",
        "conversions": "Dönüşümler",
        "cpa": "Dönüşüm Başına Maliyet",

        "performance_trend": "Performans Trendi",
        "optimization_recommendations": (
            "Optimizasyon Önerileri"
        ),
        "portfolio_allocation": (
            "Portföy Bütçe Dağılımı"
        ),
        "executive_ai_summary": (
            "Yönetici Yapay Zekâ Özeti"
        ),
        "risk_analysis": "Risk Analizi",
        "opportunity_analysis": "Fırsat Analizi",
        "model_metrics": "Model Metrikleri",
        "feature_importance": (
            "Özellik Önem Düzeyleri"
        ),
        "forecast": "Tahmin",

        "refresh_data": "Verileri Yenile",
        "run_analysis": "Analizi Başlat",
        "download_csv": "CSV İndir",
        "download_excel": "Excel İndir",

        "no_data": (
            "Seçilen dönem için veri bulunamadı."
        ),

        "platform_online": "Platform Çevrimiçi",
        "pipeline_status": "Pipeline Durumu",
        "data_source_status": "Veri Kaynağı Durumu",
        "postgres_status": "PostgreSQL Durumu",
        "llm_status": "Yapay Zekâ Modeli Durumu",

        "executive_overview_description": (
            "Reklam performansını, temel KPI'ları ve "
            "yönetici karar sinyallerini izleyin."
        ),

        "campaign_analysis_description": (
            "Kampanya sonuçlarını seçilen tarih ve "
            "karşılaştırma dönemine göre inceleyin."
        ),

        "budget_optimizer_description": (
            "Bütçe senaryolarını, önerilen dağılımları "
            "ve beklenen etkileri değerlendirin."
        ),

        "ai_insights_description": (
            "Riskleri, fırsatları ve yapay zekâ destekli "
            "aksiyon önerilerini görüntüleyin."
        ),

        "ask_ai_description": (
            "Reklam verileriniz hakkında doğal dilde "
            "sorular sorun."
        ),

        "system_status_description": (
            "Veri kaynaklarını, pipeline sürecini ve "
            "platform bileşenlerini kontrol edin."
        ),

        "read_only": (
            "Bilgilendirme: Bu ekran yalnızca analiz ve "
            "öneri sunar. Google Ads hesabınızda otomatik "
            "olarak herhangi bir değişiklik yapmaz."
        ),
    },

    "en": {
        "app_name": "Ads Budget Intelligence",
        "platform_subtitle": (
            "Enterprise Advertising Decision Platform"
        ),
        "language": "Language",
        "turkish": "Türkçe",
        "english": "English",
        "filters": "Filters",
        "date_range": "Date Range",
        "comparison": "Comparison",
        "apply_filters": "Apply Filters",
        "reset_filters": "Reset Filters",

        "today": "Today",
        "yesterday": "Yesterday",
        "this_week": "This Week",
        "last_week": "Last Week",
        "last_7_days": "Last 7 Days",
        "last_30_days": "Last 30 Days",
        "last_60_days": "Last 60 Days",
        "last_90_days": "Last 90 Days",
        "this_month": "This Month",
        "last_month": "Last Month",
        "this_quarter": "This Quarter",
        "last_quarter": "Last Quarter",
        "this_year": "Year to Date",
        "last_year": "Last Year",
        "custom_range": "Custom Date Range",

        "no_comparison": "No Comparison",
        "previous_period": "Previous Period",
        "previous_month": "Previous Month",
        "previous_quarter": "Previous Quarter",
        "previous_year": "Same Period Last Year",
        "previous_year_ytd": (
            "Previous Year-to-Date Period"
        ),
        "custom_comparison": (
            "Custom Comparison Dates"
        ),

        "comparison_length_warning": (
            "The analysis and comparison periods have "
            "different numbers of days. Consider the period "
            "length difference when interpreting percentage "
            "changes."
        ),

        "start_date": "Start Date",
        "end_date": "End Date",

        "filter_help": (
            "Select the analysis period, comparison period, "
            "and interface language."
        ),

        "selection_applies_to_all": (
            "Selections apply to all KPIs, charts, tables, "
            "and AI analyses."
        ),

        "system_mode": "System Mode",
        "online": "Online",
        "offline": "Offline",
        "deterministic_mode": "Deterministic Mode",
        "hybrid_mode": "Hybrid AI Mode",

        "executive_overview": "Executive Overview",
        "campaign_analysis": "Campaign Analysis",
        "budget_optimizer": "Budget Optimizer",
        "ai_insights": "AI Insights",
        "ask_ai": "Ask AI",
        "system_status": "System Status",
        "agency_recommendations": (
            "Agency Recommendations"
        ),
        "reports": "Reports",

        "last_update": "Last Update",
        "revenue": "Revenue",
        "ad_spend": "Ad Spend",
        "roas": "ROAS",
        "conversions": "Conversions",
        "cpa": "Cost per Acquisition",

        "performance_trend": "Performance Trend",
        "optimization_recommendations": (
            "Optimization Recommendations"
        ),
        "portfolio_allocation": (
            "Portfolio Budget Allocation"
        ),
        "executive_ai_summary": (
            "Executive AI Summary"
        ),
        "risk_analysis": "Risk Analysis",
        "opportunity_analysis": "Opportunity Analysis",
        "model_metrics": "Model Metrics",
        "feature_importance": "Feature Importance",
        "forecast": "Forecast",

        "refresh_data": "Refresh Data",
        "run_analysis": "Run Analysis",
        "download_csv": "Download CSV",
        "download_excel": "Download Excel",

        "no_data": (
            "No data was found for the selected period."
        ),

        "platform_online": "Platform Online",
        "pipeline_status": "Pipeline Status",
        "data_source_status": "Data Source Status",
        "postgres_status": "PostgreSQL Status",
        "llm_status": "AI Model Status",

        "executive_overview_description": (
            "Monitor advertising performance, core KPIs, "
            "and executive decision signals."
        ),

        "campaign_analysis_description": (
            "Review campaign results for the selected "
            "analysis and comparison periods."
        ),

        "budget_optimizer_description": (
            "Evaluate budget scenarios, recommended "
            "allocations, and expected impact."
        ),

        "ai_insights_description": (
            "Review risks, opportunities, and AI-supported "
            "action recommendations."
        ),

        "ask_ai_description": (
            "Ask natural-language questions about your "
            "advertising data."
        ),

        "system_status_description": (
            "Check data sources, the pipeline, and platform "
            "components."
        ),

        "read_only": (
            "Information: This screen provides analysis and "
            "recommendations only. It does not automatically "
            "make any changes to your Google Ads account."
        ),
    },
}


def normalize_language(
    language: str | None,
) -> str:
    """Return a supported language code."""
    if language in TRANSLATIONS:
        return language

    return DEFAULT_LANGUAGE


def translate(
    key: str,
    language: str | None = None,
    default: str | None = None,
) -> str:
    """Translate a user-interface key."""
    resolved_language = normalize_language(
        language
    )

    language_map = TRANSLATIONS[
        resolved_language
    ]

    if key in language_map:
        return language_map[key]

    if default is not None:
        return default

    return key


