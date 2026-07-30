from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)


from config.settings import TARGET_ROAS
from dashboard.components.export import render_export_buttons
from dashboard.components.tables import hide_native_dataframe_toolbar
from dashboard.layout import (
    initialize_dashboard,
    localized_text,
    render_read_only_footer,
)
from dashboard.services.executive_data import (
    get_available_date_bounds,
    get_recommendation_period,
    load_executive_data,
    recommendation_period_is_known,
)
from dashboard.services.executive_metrics import calculate_model_r2
from dashboard.services.executive_scoring import (
    build_action_summary,
    build_display_table,
    enrich_recommendations,
    get_top_opportunities,
    get_top_risks,
    localize_action,
    localize_risk,
)
from dashboard.utils import (
    get_latest_output_time,
    load_csv,
)
from src.llm.manager import get_llm_runtime_info


def text(
    language: str,
    turkish: str,
    english: str,
) -> str:
    return localized_text(
        language,
        turkish,
        english,
    )


def safe_numeric(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
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


def build_ranked_table(
    dataframe: pd.DataFrame,
    language: str,
    score_column: str,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            text(language, "Kampanya", "Campaign"): dataframe[
                "CampaignCanonical"
            ],
            text(language, "Aksiyon", "Action"): dataframe[
                "ActionCanonical"
            ].map(
                lambda value: localize_action(
                    value,
                    language,
                )
            ),
            text(language, "Risk", "Risk"): dataframe[
                "RiskLevelCanonical"
            ].map(
                lambda value: localize_risk(
                    value,
                    language,
                )
            ),
            text(language, "Tahmini ROAS", "Predicted ROAS"): dataframe[
                "PredictedROASCanonical"
            ],
            text(language, "Bütçe Değişimi %", "Budget Change %"): dataframe[
                "BudgetChangePctCanonical"
            ],
            text(language, "Skor", "Score"): dataframe[
                score_column
            ],
            text(language, "Gerekçe", "Rationale"): dataframe.apply(
                lambda row: localized_reason(row, language),
                axis=1,
            ),
        }
    )


def localized_reason(row: pd.Series, language: str) -> str:
    action = str(row.get("ActionCanonical", "unknown"))
    predicted_roas = float(row.get("PredictedROASCanonical", 0.0) or 0.0)
    current_spend = float(row.get("CurrentSpendCanonical", 0.0) or 0.0)

    if current_spend <= 0 or action == "review":
        return text(
            language,
            "Aktif harcama bulunmadığı için manuel inceleme gerekiyor.",
            "Manual review is required because no active spend was found.",
        )
    if action == "increase" and predicted_roas < TARGET_ROAS:
        return text(
            language,
            "Ölçekleme potansiyeli var; ancak tahmini ROAS hedefin altında. "
            "Artış kontrollü uygulanmalı.",
            "Scaling potential exists, but predicted ROAS is below target. "
            "Any increase should be controlled.",
        )
    if action == "increase":
        return text(
            language,
            "Tahmini performans bütçe artışını destekliyor; kampanya "
            "kontrollü biçimde ölçeklenebilir.",
            "Predicted performance supports a controlled budget increase.",
        )
    if action == "reduce" and predicted_roas >= TARGET_ROAS:
        return text(
            language,
            "ROAS güçlü olsa da model daha düşük bütçeyle verimliliğin "
            "korunabileceğini öngörüyor. Bu bir tasarruf fırsatıdır.",
            "Although ROAS is strong, the model expects efficiency to be "
            "maintained with less budget. This is a savings opportunity.",
        )
    if action == "reduce":
        return text(
            language,
            "Tahmini getiri hedefin altında veya kampanya mevcut düzeyde "
            "fazla bütçelenmiş görünüyor.",
            "Predicted return is below target or the campaign appears "
            "overfunded.",
        )
    return text(
        language,
        "Model mevcut bütçeye yakın seviyenin korunmasını destekliyor.",
        "The model supports keeping the budget near its current level.",
    )


def build_rationale_table(
    dataframe: pd.DataFrame,
    language: str,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            text(language, "Kampanya", "Campaign"):
                dataframe["CampaignCanonical"],
            text(language, "Aksiyon", "Action"):
                dataframe["ActionCanonical"].map(
                    lambda value: localize_action(value, language)
                ),
            text(language, "Risk", "Risk"):
                dataframe["RiskLevelCanonical"].map(
                    lambda value: localize_risk(value, language)
                ),
            text(language, "Karar Gerekçesi", "Decision Rationale"):
                dataframe.apply(
                    lambda row: localized_reason(row, language),
                    axis=1,
                ),
            text(language, "Mevcut Harcama", "Current Spend"):
                dataframe["CurrentSpendCanonical"],
            text(language, "Önerilen Bütçe", "Recommended Budget"):
                dataframe["RecommendedBudgetCanonical"],
            text(language, "Bütçe Değişimi %", "Budget Change %"):
                dataframe["BudgetChangePctCanonical"],
            text(language, "Tahmini ROAS", "Predicted ROAS"):
                dataframe["PredictedROASCanonical"],
            text(language, "Öneri Güveni", "Recommendation Confidence"):
                dataframe["ConfidenceScoreCanonical"] * 100,
        }
    )


def build_feature_chart(
    feature_data: pd.DataFrame,
    language: str,
):
    required = {
        "Feature",
        "Importance",
        "Model",
    }

    if feature_data.empty or not required.issubset(
        feature_data.columns
    ):
        return None

    chart_data = feature_data[
        [
            "Feature",
            "Importance",
            "Model",
        ]
    ].copy()

    chart_data["Importance"] = safe_numeric(
        chart_data,
        "Importance",
    )

    chart_data = (
        chart_data.sort_values(
            "Importance",
            ascending=False,
        )
        .groupby(
            "Model",
            group_keys=False,
        )
        .head(8)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    figure = px.bar(
        chart_data,
        x="Importance",
        y="Feature",
        color="Model",
        orientation="h",
        barmode="group",
        labels={
            "Importance": text(
                language,
                "Önem Düzeyi",
                "Importance",
            ),
            "Feature": text(
                language,
                "Özellik",
                "Feature",
            ),
            "Model": text(
                language,
                "Model",
                "Model",
            ),
        },
    )

    figure.update_layout(
        height=500,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    return figure


def build_model_table(
    model_metrics: pd.DataFrame,
    language: str,
) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()

    result = model_metrics.copy()

    for column in [
        "MAE",
        "RMSE",
        "R2",
        "TrainRows",
        "TestRows",
    ]:
        if column in result.columns:
            result[column] = safe_numeric(
                result,
                column,
            )

    if "R2" in result.columns:
        result["R2"] = result["R2"] * 100

    rename_map = {
        "Model": text(language, "Model", "Model"),
        "MAE": "MAE",
        "RMSE": "RMSE",
        "R2": text(language, "R² (%)", "R² (%)"),
        "TrainRows": text(
            language,
            "Eğitim Satırı",
            "Training Rows",
        ),
        "TestRows": text(
            language,
            "Test Satırı",
            "Test Rows",
        ),
    }

    return result.rename(
        columns=rename_map
    )


data = load_executive_data()
feature_importance = load_csv(
    "ads_feature_importance.csv"
)
available_start, available_end = get_available_date_bounds(
    data.daily
)

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=(
        "Yapay Zeka Analizleri"
        if initial_language == "tr"
        else "AI Insights"
    ),
    page_icon="🧠",
    title=(
        "Yapay Zeka Analizleri"
        if initial_language == "tr"
        else "AI Insights"
    ),
    subtitle=(
        "Model sonuçlarını, riskleri, fırsatları ve karar gerekçelerini inceleyin."
        if initial_language == "tr"
        else
        "Review model results, risks, opportunities, and decision rationale."
    ),
    reference_date=available_end,
)

language = context.language
hide_native_dataframe_toolbar()

enriched = enrich_recommendations(
    data.recommendations,
    target_roas=TARGET_ROAS,
)

summary = build_action_summary(
    enriched
)

opportunities = get_top_opportunities(
    enriched,
    limit=5,
)

risks = get_top_risks(
    enriched,
    limit=5,
)

model_r2 = calculate_model_r2(
    data.model_metrics
)

runtime = get_llm_runtime_info()
llm_ready = bool(runtime.get("ready"))
provider = str(runtime.get("provider") or "-")
model_name = str(runtime.get("model") or "-")


st.subheader(
    text(
        language,
        "Analiz Çalışma Modu",
        "Analysis Runtime",
    )
)

mode_col, accuracy_col = st.columns(2)

with mode_col:
    st.metric(
        text(language, "Çalışma Modu", "Runtime Mode"),
        (
            text(language, "Hibrit AI", "Hybrid AI")
            if llm_ready
            else text(
                language,
                "Deterministik",
                "Deterministic",
            )
        ),
    )

with accuracy_col:
    st.metric(
        text(language, "Ortalama Model R²", "Average Model R²"),
        f"%{model_r2:.1f}" if model_r2 is not None else "-",
    )

provider_col, model_col = st.columns(2)

with provider_col:
    st.metric(
        text(language, "LLM Sağlayıcısı", "LLM Provider"),
        (
            provider.title()
            if llm_ready
            else text(
                language,
                "Bağlı Değil",
                "Not Connected",
            )
        ),
    )

with model_col:
    st.metric(
        text(language, "LLM Modeli", "LLM Model"),
        (
            model_name
            if llm_ready
            else "-"
        ),
    )

if llm_ready:
    st.success(
        text(
            language,
            "LLM bağlantısı hazır. Kural tabanlı sonuçlar, model çıktıları "
            "ve LLM açıklamaları birlikte kullanılabilir.",
            "The LLM connection is ready. Rule-based results, model outputs, "
            "and LLM explanations can be used together.",
        )
    )
else:
    st.info(
        text(
            language,
            "API anahtarı veya model yapılandırması hazır olmadığı için bu "
            "sayfa deterministik modda çalışıyor. Risk, fırsat, skor ve bütçe "
            "hesapları kullanılabilir; metinler güvenli kural tabanlı "
            "açıklamalardır.",
            "This page is running in deterministic mode because the API key "
            "or model configuration is not ready. Risk, opportunity, score, "
            "and budget calculations remain available; commentary uses safe "
            "rule-based explanations.",
        )
    )

st.info(
    text(
        language,
        "Bu sayfadaki model metrikleri, riskler, fırsatlar ve öneriler son "
        "pipeline çalışmasının kayıtlı çıktılarıdır. Üstteki tarih seçimi "
        "bu çıktıları yeniden eğitmez veya yeniden üretmez.",
        "Model metrics, risks, opportunities, and recommendations are saved "
        "outputs from the latest pipeline run. The date selection above "
        "does not retrain or regenerate them.",
    )
)

recommendation_start, recommendation_end = (
    get_recommendation_period(
        data.recommendations
    )
)

recommendation_matches_selection = (
    recommendation_period_is_known(
        data.recommendations
    )
    and recommendation_start == context.filters.start_date
    and recommendation_end == context.filters.end_date
)

if not recommendation_matches_selection:
    st.warning(
        text(
            language,
            (
                "Ekrandaki kayıtlı model çıktısı seçilen döneme ait değil. "
                "Tarih bölümünü açıp “Seçilen Dönemi Analiz Et” düğmesine "
                "basın. Doğru dönem üretilmeden eski risk ve öneriler "
                "gösterilmeyecektir."
            ),
            (
                "The saved model output does not belong to the selected "
                "period. Open the Date section and click “Analyze Selected "
                "Period”. Stale risks and recommendations will not be shown "
                "until the correct period is generated."
            ),
        ),
        icon="⚠️",
    )
    st.stop()

st.subheader(
    text(language, "Yönetici Karar Özeti", "Executive Decision Summary")
)
st.write(
    text(
        language,
        f"Toplam {len(enriched)} kampanyanın {summary.increase_count} "
        f"tanesinde bütçe artışı, {summary.reduce_count} tanesinde bütçe "
        f"azaltımı, {summary.maintain_count} tanesinde bütçeyi koruma ve "
        f"{summary.review_count} tanesinde manuel inceleme öneriliyor. "
        f"{summary.high_risk_count} kampanya yüksek riskli; "
        f"{summary.insufficient_data_count} kampanyada aktif veri yetersiz.",
        f"Across {len(enriched)} campaigns, increases are recommended for "
        f"{summary.increase_count}, reductions for {summary.reduce_count}, "
        f"maintenance for {summary.maintain_count}, and manual review for "
        f"{summary.review_count}. {summary.high_risk_count} campaigns are "
        f"high risk and {summary.insufficient_data_count} lack active data.",
    )
)


st.divider()
st.subheader(
    text(
        language,
        "Karar Sinyalleri",
        "Decision Signals",
    )
)

signal_columns = st.columns(6)
signal_values = [
    (
        text(language, "Bütçe Artır", "Increase"),
        summary.increase_count,
    ),
    (
        text(language, "Bütçe Azalt", "Reduce"),
        summary.reduce_count,
    ),
    (
        text(language, "Bütçeyi Koru", "Maintain"),
        summary.maintain_count,
    ),
    (
        text(language, "İncele", "Review"),
        summary.review_count,
    ),
    (
        text(language, "Yüksek Risk", "High Risk"),
        summary.high_risk_count,
    ),
    (
        text(language, "Veri Yetersiz", "Insufficient Data"),
        summary.insufficient_data_count,
    ),
]

for column, (label, value) in zip(
    signal_columns,
    signal_values,
):
    with column:
        st.metric(label, value)


st.divider()
left_column, right_column = st.columns(2)

opportunity_table = build_ranked_table(
    opportunities,
    language,
    "OpportunityScoreCanonical",
)

risk_table = build_ranked_table(
    risks,
    language,
    "RiskScoreCanonical",
)

with left_column:
    st.subheader(
        text(
            language,
            "İlk 5 Fırsat",
            "Top 5 Opportunities",
        )
    )

    if opportunity_table.empty:
        st.info(
            text(
                language,
                "Fırsat verisi bulunamadı.",
                "No opportunity data was found.",
            )
        )
    else:
        st.dataframe(
            opportunity_table,
            hide_index=True,
            width="stretch",
            column_config={
                opportunity_table.columns[3]: st.column_config.NumberColumn(
                    format="%.2fx"
                ),
                opportunity_table.columns[4]: st.column_config.NumberColumn(
                    format="%+.1f%%"
                ),
                opportunity_table.columns[5]: st.column_config.NumberColumn(
                    format="%.1f"
                ),
            },
        )

with right_column:
    st.subheader(
        text(
            language,
            "İlk 5 Risk",
            "Top 5 Risks",
        )
    )

    if risk_table.empty:
        st.info(
            text(
                language,
                "Risk verisi bulunamadı.",
                "No risk data was found.",
            )
        )
    else:
        st.dataframe(
            risk_table,
            hide_index=True,
            width="stretch",
            column_config={
                risk_table.columns[3]: st.column_config.NumberColumn(
                    format="%.2fx"
                ),
                risk_table.columns[4]: st.column_config.NumberColumn(
                    format="%+.1f%%"
                ),
                risk_table.columns[5]: st.column_config.NumberColumn(
                    format="%.1f"
                ),
            },
        )


st.divider()
st.subheader(
    text(
        language,
        "Model Performansı",
        "Model Performance",
    )
)

model_table = build_model_table(
    data.model_metrics,
    language,
)

if model_table.empty:
    st.info(
        text(
            language,
            "Model doğrulama metrikleri bulunamadı.",
            "Model validation metrics were not found.",
        )
    )
else:
    st.dataframe(
        model_table,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(
                format="%.2f"
            ),
            "RMSE": st.column_config.NumberColumn(
                format="%.2f"
            ),
            text(
                language,
                "R² (%)",
                "R² (%)",
            ): st.column_config.NumberColumn(
                format="%.1f%%"
            ),
        },
    )

    st.caption(
        text(
            language,
            "R², modelin test verisindeki değişimin ne kadarını açıkladığını "
            "gösterir. MAE ve RMSE tahmin hatasını ölçer; daha düşük değer "
            "daha iyidir.",
            "R² shows how much variation the model explains on test data. "
            "MAE and RMSE measure prediction error; lower values are better.",
        )
    )


st.subheader(
    text(
        language,
        "Özellik Önem Düzeyleri",
        "Feature Importance",
    )
)

feature_chart = build_feature_chart(
    feature_importance,
    language,
)

if feature_chart is None:
    st.info(
        text(
            language,
            "Özellik önem verisi bulunamadı.",
            "Feature importance data was not found.",
        )
    )
else:
    st.plotly_chart(
        feature_chart,
        width="stretch",
    )

    st.caption(
        text(
            language,
            "Önem düzeyi nedensellik göstermez; modelin tahmin üretirken "
            "hangi değişkenlerden daha fazla yararlandığını gösterir.",
            "Importance does not imply causality; it shows which variables "
            "the model relied on more when producing predictions.",
        )
    )


st.divider()
st.subheader(
    text(
        language,
        "Kampanya Karar Gerekçeleri",
        "Campaign Decision Rationale",
    )
)

display_recommendations = build_display_table(
    enriched,
    language,
)

if display_recommendations.empty:
    st.info(
        text(
            language,
            "Öneri verisi bulunamadı.",
            "Recommendation data was not found.",
        )
    )
else:
    selected_action = st.selectbox(
        text(
            language,
            "Aksiyon Filtresi",
            "Action Filter",
        ),
        options=[
            "all",
            "increase",
            "reduce",
            "maintain",
            "review",
        ],
        format_func=lambda value: (
            text(
                language,
                "Tüm aksiyonlar",
                "All actions",
            )
            if value == "all"
            else localize_action(
                value,
                language,
            )
        ),
    )

    filtered_enriched = enriched.copy()

    if selected_action != "all":
        filtered_enriched = filtered_enriched.loc[
            filtered_enriched[
                "ActionCanonical"
            ].eq(selected_action)
        ].copy()

    rationale_table = build_rationale_table(
        filtered_enriched,
        language,
    )

    st.dataframe(
        rationale_table,
        hide_index=True,
        width="stretch",
        column_config={
            rationale_table.columns[4]: st.column_config.NumberColumn(
                format="₺%.2f"
            ),
            rationale_table.columns[5]: st.column_config.NumberColumn(
                format="₺%.2f"
            ),
            rationale_table.columns[6]: st.column_config.NumberColumn(
                format="%+.1f%%"
            ),
            rationale_table.columns[7]: st.column_config.NumberColumn(
                format="%.2fx"
            ),
            rationale_table.columns[8]: st.column_config.NumberColumn(
                format="%.0f%%"
            ),
        },
    )


st.divider()
st.subheader(
    text(
        language,
        "AI Analiz Raporunu Dışa Aktar",
        "Export AI Insights Report",
    )
)

render_export_buttons(
    csv_dataframe=display_recommendations,
    excel_sheets={
        "Recommendations": display_recommendations,
        "Opportunities": opportunity_table,
        "Risks": risk_table,
        "Model Metrics": model_table,
        "Feature Importance": feature_importance,
    },
    file_name="ai_insights_report",
    language=language,
    key_prefix="ai_insights",
)

st.caption(
    text(
        language,
        f"Son çıktı: {get_latest_output_time()}",
        f"Latest output: {get_latest_output_time()}",
    )
)

render_read_only_footer(language)
