from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)


from dashboard.components.export import render_export_buttons
from dashboard.components.tables import (
    hide_native_dataframe_toolbar,
)
from dashboard.layout import (
    initialize_dashboard,
    localized_text,
    render_read_only_footer,
)
from dashboard.services.executive_data import (
    get_available_date_bounds,
    get_recommendation_period,
    recommendation_period_is_known,
)
from dashboard.services.executive_scoring import (
    enrich_recommendations,
    localize_action,
    localize_risk,
)
from dashboard.utils import (
    get_latest_output_time,
    load_csv,
)


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a safe numeric series for optimizer calculations."""

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


def prepare_optimizer_data(
    recommendations: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Build one canonical campaign budget dataset."""

    enriched = enrich_recommendations(
        recommendations
    )

    if enriched.empty:
        return enriched

    result = enriched.copy()

    portfolio_budget = pd.DataFrame()

    if (
        not portfolio.empty
        and "Campaign" in portfolio.columns
        and "OptimizedPortfolioBudget" in portfolio.columns
    ):
        portfolio_budget = portfolio[
            [
                "Campaign",
                "OptimizedPortfolioBudget",
            ]
        ].copy()

        portfolio_budget[
            "OptimizedPortfolioBudget"
        ] = numeric_series(
            portfolio_budget,
            "OptimizedPortfolioBudget",
        )

    if not portfolio_budget.empty:
        result = result.merge(
            portfolio_budget,
            left_on="CampaignCanonical",
            right_on="Campaign",
            how="left",
            validate="one_to_one",
        )
    else:
        result["OptimizedPortfolioBudget"] = np.nan

    result["PortfolioBudgetCanonical"] = (
        pd.to_numeric(
            result["OptimizedPortfolioBudget"],
            errors="coerce",
        )
        .fillna(
            result["RecommendedBudgetCanonical"]
        )
        .astype("float64")
    )

    return result


def build_budget_plan(
    optimizer_data: pd.DataFrame,
    plan_mode: str,
    custom_total_budget: float | None,
) -> pd.DataFrame:
    """Create campaign budgets for the selected planning mode."""

    result = optimizer_data.copy()

    if plan_mode == "model":
        result["PlannedBudget"] = result[
            "RecommendedBudgetCanonical"
        ]
    elif plan_mode == "portfolio":
        result["PlannedBudget"] = result[
            "PortfolioBudgetCanonical"
        ]
    else:
        weights = (
            result["PortfolioBudgetCanonical"]
            .clip(lower=0.0)
        )

        if float(weights.sum()) <= 0:
            weights = (
                result["CurrentSpendCanonical"]
                .clip(lower=0.0)
            )

        if float(weights.sum()) <= 0:
            weights = pd.Series(
                1.0,
                index=result.index,
            )

        requested_total = max(
            float(custom_total_budget or 0.0),
            0.0,
        )

        result["PlannedBudget"] = (
            weights
            .div(float(weights.sum()))
            .mul(requested_total)
        )

    result["PlannedBudget"] = (
        numeric_series(
            result,
            "PlannedBudget",
        )
        .clip(lower=0.0)
    )

    result["PlannedBudgetChange"] = (
        result["PlannedBudget"]
        - result["CurrentSpendCanonical"]
    )

    result["PlannedBudgetChangePct"] = (
        result["PlannedBudgetChange"]
        .div(
            result["CurrentSpendCanonical"].where(
                result["CurrentSpendCanonical"] > 0
            )
        )
        .mul(100)
        .fillna(0.0)
    )

    return result


def interpolate_campaign_predictions(
    plan: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate plan outcomes from each campaign's ML scenario curve."""

    result = plan.copy()

    prediction_columns = {
        "PredictedConversions": (
            "PlannedPredictedConversions"
        ),
        "PredictedRevenue": "PlannedPredictedRevenue",
        "PredictedProfit": "PlannedPredictedProfit",
        "PredictedROAS": "PlannedPredictedROAS",
    }

    for target in prediction_columns.values():
        result[target] = 0.0

    result["ScenarioMinBudget"] = 0.0
    result["ScenarioMaxBudget"] = 0.0
    result["OutsideScenarioRange"] = False

    if scenarios.empty or "Campaign" not in scenarios.columns:
        return result

    scenario_data = scenarios.copy()
    scenario_data["ScenarioSpend"] = numeric_series(
        scenario_data,
        "ScenarioSpend",
    )

    for source in prediction_columns:
        scenario_data[source] = numeric_series(
            scenario_data,
            source,
        )

    zero_budget_scenarios = (
        scenario_data["ScenarioSpend"] <= 0
    )
    scenario_data.loc[
        zero_budget_scenarios,
        list(prediction_columns.keys()),
    ] = 0.0

    scenario_groups = {
        str(campaign): group.copy()
        for campaign, group in scenario_data.groupby(
            "Campaign",
            dropna=False,
        )
    }

    for index, row in result.iterrows():
        campaign = str(row["CampaignCanonical"])
        campaign_scenarios = scenario_groups.get(campaign)

        if campaign_scenarios is None:
            continue

        curve = (
            campaign_scenarios.sort_values(
                "ScenarioSpend"
            )
            .drop_duplicates(
                subset=["ScenarioSpend"],
                keep="last",
            )
        )

        if curve.empty:
            continue

        x_values = curve["ScenarioSpend"].to_numpy(
            dtype=float
        )
        planned_budget = float(row["PlannedBudget"])
        minimum_budget = float(x_values.min())
        maximum_budget = float(x_values.max())

        result.at[index, "ScenarioMinBudget"] = (
            minimum_budget
        )
        result.at[index, "ScenarioMaxBudget"] = (
            maximum_budget
        )
        result.at[index, "OutsideScenarioRange"] = (
            planned_budget < minimum_budget * 0.90
            or planned_budget > maximum_budget * 1.10
        )

        for source, target in prediction_columns.items():
            y_values = curve[source].to_numpy(
                dtype=float
            )

            if len(x_values) == 1:
                predicted_value = float(y_values[0])
            else:
                predicted_value = float(
                    np.interp(
                        planned_budget,
                        x_values,
                        y_values,
                    )
                )

            result.at[index, target] = predicted_value

    result["PlannedPredictedROAS"] = (
        result["PlannedPredictedRevenue"]
        .div(
            result["PlannedBudget"].where(
                result["PlannedBudget"] > 0
            )
        )
        .fillna(0.0)
    )

    zero_plan_mask = result["PlannedBudget"] <= 0
    result.loc[
        zero_plan_mask,
        [
            "PlannedPredictedConversions",
            "PlannedPredictedRevenue",
            "PlannedPredictedProfit",
            "PlannedPredictedROAS",
        ],
    ] = 0.0

    return result


def build_plan_table(
    plan: pd.DataFrame,
    language: str,
) -> pd.DataFrame:
    """Build a localized plan table with numeric values preserved."""

    if plan.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            localized_text(
                language,
                "Kampanya",
                "Campaign",
            ): plan["CampaignCanonical"],
            localized_text(
                language,
                "Kampanya Türü",
                "Campaign Type",
            ): plan["CampaignTypeCanonical"].replace(
                {
                    "Brand": localized_text(
                        language,
                        "Marka",
                        "Brand",
                    ),
                    "Generic": localized_text(
                        language,
                        "Genel",
                        "Generic",
                    ),
                }
            ),
            localized_text(
                language,
                "Mevcut Günlük Harcama",
                "Current Daily Spend",
            ): plan["CurrentSpendCanonical"],
            localized_text(
                language,
                "Planlanan Günlük Bütçe",
                "Planned Daily Budget",
            ): plan["PlannedBudget"],
            localized_text(
                language,
                "Bütçe Değişimi (%)",
                "Budget Change (%)",
            ): plan["PlannedBudgetChangePct"],
            localized_text(
                language,
                "Tahmini Günlük Gelir",
                "Predicted Daily Revenue",
            ): plan["PlannedPredictedRevenue"],
            localized_text(
                language,
                "Tahmini Günlük Kâr",
                "Predicted Daily Profit",
            ): plan["PlannedPredictedProfit"],
            localized_text(
                language,
                "Tahmini ROAS",
                "Predicted ROAS",
            ): plan["PlannedPredictedROAS"],
            localized_text(
                language,
                "Aksiyon",
                "Action",
            ): plan["ActionCanonical"].map(
                lambda value: localize_action(
                    value,
                    language,
                )
            ),
            localized_text(
                language,
                "Risk",
                "Risk",
            ): plan["RiskLevelCanonical"].map(
                lambda value: localize_risk(
                    value,
                    language,
                )
            ),
            localized_text(
                language,
                "Senaryo Aralığı",
                "Scenario Range",
            ): np.where(
                plan["OutsideScenarioRange"],
                localized_text(
                    language,
                    "Aralık Dışı",
                    "Outside Range",
                ),
                localized_text(
                    language,
                    "Uygun",
                    "Within Range",
                ),
            ),
        }
    )


def plan_table_column_config(
    language: str,
) -> dict[str, object]:
    """Return optimizer table formats."""

    labels = {
        "current": localized_text(
            language,
            "Mevcut Günlük Harcama",
            "Current Daily Spend",
        ),
        "planned": localized_text(
            language,
            "Planlanan Günlük Bütçe",
            "Planned Daily Budget",
        ),
        "change": localized_text(
            language,
            "Bütçe Değişimi (%)",
            "Budget Change (%)",
        ),
        "revenue": localized_text(
            language,
            "Tahmini Günlük Gelir",
            "Predicted Daily Revenue",
        ),
        "profit": localized_text(
            language,
            "Tahmini Günlük Kâr",
            "Predicted Daily Profit",
        ),
        "roas": localized_text(
            language,
            "Tahmini ROAS",
            "Predicted ROAS",
        ),
    }

    return {
        labels["current"]: st.column_config.NumberColumn(
            labels["current"],
            format="₺%.2f",
        ),
        labels["planned"]: st.column_config.NumberColumn(
            labels["planned"],
            format="₺%.2f",
        ),
        labels["change"]: st.column_config.NumberColumn(
            labels["change"],
            format="%+.1f%%",
        ),
        labels["revenue"]: st.column_config.NumberColumn(
            labels["revenue"],
            format="₺%.2f",
        ),
        labels["profit"]: st.column_config.NumberColumn(
            labels["profit"],
            format="₺%.2f",
        ),
        labels["roas"]: st.column_config.NumberColumn(
            labels["roas"],
            format="%.2fx",
        ),
    }


def render_plan_kpis(
    plan: pd.DataFrame,
    language: str,
) -> None:
    """Render budget-plan totals."""

    current_total = float(
        plan["CurrentSpendCanonical"].sum()
    )
    planned_total = float(
        plan["PlannedBudget"].sum()
    )
    predicted_revenue = float(
        plan["PlannedPredictedRevenue"].sum()
    )
    predicted_profit = float(
        plan["PlannedPredictedProfit"].sum()
    )
    predicted_conversions = float(
        plan["PlannedPredictedConversions"].sum()
    )
    portfolio_roas = (
        predicted_revenue / planned_total
        if planned_total > 0
        else 0.0
    )

    budget_change_pct = (
        (planned_total - current_total)
        / current_total
        * 100
        if current_total > 0
        else 0.0
    )

    first_row = st.columns(3)
    first_row[0].metric(
        localized_text(
            language,
            "Mevcut Toplam Günlük Harcama",
            "Current Total Daily Spend",
        ),
        f"₺{current_total:,.2f}",
    )
    first_row[1].metric(
        localized_text(
            language,
            "Planlanan Toplam Günlük Bütçe",
            "Planned Total Daily Budget",
        ),
        f"₺{planned_total:,.2f}",
        f"{budget_change_pct:+.1f}%",
    )
    first_row[2].metric(
        localized_text(
            language,
            "Portföy Tahmini ROAS",
            "Portfolio Predicted ROAS",
        ),
        f"{portfolio_roas:.2f}x",
    )

    second_row = st.columns(3)
    second_row[0].metric(
        localized_text(
            language,
            "Tahmini Günlük Gelir",
            "Predicted Daily Revenue",
        ),
        f"₺{predicted_revenue:,.2f}",
    )
    second_row[1].metric(
        localized_text(
            language,
            "Tahmini Günlük Kâr",
            "Predicted Daily Profit",
        ),
        f"₺{predicted_profit:,.2f}",
    )
    second_row[2].metric(
        localized_text(
            language,
            "Tahmini Günlük Dönüşüm",
            "Predicted Daily Conversions",
        ),
        f"{predicted_conversions:,.2f}",
    )


def render_allocation_chart(
    plan: pd.DataFrame,
    language: str,
) -> None:
    """Compare current and planned campaign budgets."""

    chart_data = (
        plan.sort_values(
            "PlannedBudget",
            ascending=False,
        )
        .head(10)
        .sort_values(
            "PlannedBudget",
            ascending=True,
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            y=chart_data["CampaignCanonical"],
            x=chart_data["CurrentSpendCanonical"],
            name=localized_text(
                language,
                "Mevcut Günlük Harcama",
                "Current Daily Spend",
            ),
            orientation="h",
            marker_color="#7dd3fc",
            hovertemplate=(
                "%{y}<br>₺%{x:,.2f}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Bar(
            y=chart_data["CampaignCanonical"],
            x=chart_data["PlannedBudget"],
            name=localized_text(
                language,
                "Planlanan Günlük Bütçe",
                "Planned Daily Budget",
            ),
            orientation="h",
            marker_color="#087ccc",
            hovertemplate=(
                "%{y}<br>₺%{x:,.2f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        barmode="group",
        height=540,
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=20,
        ),
        xaxis_title=localized_text(
            language,
            "Günlük Bütçe (₺)",
            "Daily Budget (₺)",
        ),
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
        key="budget_optimizer_allocation_chart",
    )


def render_scenario_explorer(
    scenarios: pd.DataFrame,
    language: str,
) -> None:
    """Render one campaign's source ML scenarios."""

    st.subheader(
        localized_text(
            language,
            "Kampanya Senaryo Gezgini",
            "Campaign Scenario Explorer",
        )
    )

    if scenarios.empty or "Campaign" not in scenarios.columns:
        st.info(
            localized_text(
                language,
                "Senaryo verisi bulunmuyor.",
                "Scenario data is not available.",
            )
        )
        return

    campaign_options = sorted(
        scenarios["Campaign"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_campaign = st.selectbox(
        localized_text(
            language,
            "Kampanya",
            "Campaign",
        ),
        campaign_options,
        key="budget_optimizer_scenario_campaign",
    )

    campaign_scenarios = (
        scenarios[
            scenarios["Campaign"].astype(str)
            == selected_campaign
        ]
        .copy()
        .sort_values("ScenarioSpend")
    )

    for column in [
        "ScenarioSpend",
        "PredictedRevenue",
        "PredictedProfit",
        "PredictedROAS",
    ]:
        campaign_scenarios[column] = numeric_series(
            campaign_scenarios,
            column,
        )

    zero_budget_scenarios = (
        campaign_scenarios["ScenarioSpend"] <= 0
    )
    campaign_scenarios.loc[
        zero_budget_scenarios,
        [
            "PredictedConversions",
            "PredictedRevenue",
            "PredictedProfit",
            "PredictedROAS",
        ],
    ] = 0.0

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=campaign_scenarios["ScenarioSpend"],
            y=campaign_scenarios["PredictedRevenue"],
            mode="lines+markers",
            name=localized_text(
                language,
                "Tahmini Günlük Gelir",
                "Predicted Daily Revenue",
            ),
            line=dict(color="#38bdf8"),
            hovertemplate=(
                "Bütçe: ₺%{x:,.2f}<br>"
                "Gelir: ₺%{y:,.2f}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=campaign_scenarios["ScenarioSpend"],
            y=campaign_scenarios["PredictedProfit"],
            mode="lines+markers",
            name=localized_text(
                language,
                "Tahmini Günlük Kâr",
                "Predicted Daily Profit",
            ),
            line=dict(color="#22c55e"),
            hovertemplate=(
                "Bütçe: ₺%{x:,.2f}<br>"
                "Kâr: ₺%{y:,.2f}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=campaign_scenarios["ScenarioSpend"],
            y=campaign_scenarios["PredictedROAS"],
            mode="lines+markers",
            name="ROAS",
            yaxis="y2",
            line=dict(color="#f59e0b"),
            hovertemplate=(
                "Bütçe: ₺%{x:,.2f}<br>"
                "ROAS: %{y:.2f}x<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=470,
        margin=dict(
            l=10,
            r=10,
            t=70,
            b=20,
        ),
        xaxis_title=localized_text(
            language,
            "Günlük Senaryo Bütçesi (₺)",
            "Daily Scenario Budget (₺)",
        ),
        yaxis=dict(
            title=localized_text(
                language,
                "Gelir / Kâr (₺)",
                "Revenue / Profit (₺)",
            )
        ),
        yaxis2=dict(
            title="ROAS",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
        key="budget_optimizer_scenario_chart",
    )

    scenario_table = campaign_scenarios[
        [
            "ScenarioFactor",
            "ScenarioSpend",
            "PredictedConversions",
            "PredictedRevenue",
            "PredictedProfit",
            "PredictedROAS",
        ]
    ].rename(
        columns={
            "ScenarioFactor": localized_text(
                language,
                "Senaryo Çarpanı",
                "Scenario Factor",
            ),
            "ScenarioSpend": localized_text(
                language,
                "Günlük Senaryo Bütçesi",
                "Daily Scenario Budget",
            ),
            "PredictedConversions": localized_text(
                language,
                "Tahmini Günlük Dönüşüm",
                "Predicted Daily Conversions",
            ),
            "PredictedRevenue": localized_text(
                language,
                "Tahmini Günlük Gelir",
                "Predicted Daily Revenue",
            ),
            "PredictedProfit": localized_text(
                language,
                "Tahmini Günlük Kâr",
                "Predicted Daily Profit",
            ),
            "PredictedROAS": localized_text(
                language,
                "Tahmini ROAS",
                "Predicted ROAS",
            ),
        }
    )

    st.dataframe(
        scenario_table,
        width="stretch",
        hide_index=True,
        column_config={
            localized_text(
                language,
                "Senaryo Çarpanı",
                "Scenario Factor",
            ): st.column_config.NumberColumn(
                format="%.2fx"
            ),
            localized_text(
                language,
                "Günlük Senaryo Bütçesi",
                "Daily Scenario Budget",
            ): st.column_config.NumberColumn(
                format="₺%.2f"
            ),
            localized_text(
                language,
                "Tahmini Günlük Dönüşüm",
                "Predicted Daily Conversions",
            ): st.column_config.NumberColumn(
                format="%.2f"
            ),
            localized_text(
                language,
                "Tahmini Günlük Gelir",
                "Predicted Daily Revenue",
            ): st.column_config.NumberColumn(
                format="₺%.2f"
            ),
            localized_text(
                language,
                "Tahmini Günlük Kâr",
                "Predicted Daily Profit",
            ): st.column_config.NumberColumn(
                format="₺%.2f"
            ),
            localized_text(
                language,
                "Tahmini ROAS",
                "Predicted ROAS",
            ): st.column_config.NumberColumn(
                format="%.2fx"
            ),
        },
        key="budget_optimizer_scenario_table",
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

daily_df = load_csv("ads_daily_fact.csv")
recommendations_df = load_csv(
    "ads_budget_optimization_recommendations.csv"
)
scenarios_df = load_csv("ads_budget_scenarios.csv")
portfolio_df = load_csv(
    "ads_portfolio_budget_allocation.csv"
)

available_start, available_end = (
    get_available_date_bounds(daily_df)
)


# ---------------------------------------------------------
# INITIALIZE
# ---------------------------------------------------------

initial_language = st.session_state.get(
    "dashboard_language",
    "tr",
)

context = initialize_dashboard(
    page_title=localized_text(
        initial_language,
        "Bütçe Optimizasyonu",
        "Budget Optimizer",
    ),
    page_icon="💰",
    title=localized_text(
        initial_language,
        "Bütçe Optimizasyonu",
        "Budget Optimizer",
    ),
    subtitle=localized_text(
        initial_language,
        (
            "Kampanya bütçe senaryolarını karşılaştırın, "
            "portföy dağılımını planlayın ve beklenen "
            "sonuçları inceleyin."
        ),
        (
            "Compare campaign budget scenarios, plan "
            "portfolio allocation, and review expected "
            "outcomes."
        ),
    ),
    reference_date=available_end,
)

language = context.language

hide_native_dataframe_toolbar()


# ---------------------------------------------------------
# VALIDATE OUTPUTS
# ---------------------------------------------------------

if recommendations_df.empty:
    st.error(
        localized_text(
            language,
            (
                "Bütçe optimizasyon önerisi bulunamadı. "
                "Önce veri pipeline'ını çalıştırın."
            ),
            (
                "Budget optimization recommendations "
                "were not found. Run the data pipeline first."
            ),
        )
    )
    render_read_only_footer(language)
    st.stop()

recommendation_start, recommendation_end = (
    get_recommendation_period(
        recommendations_df
    )
)

recommendation_matches_selection = (
    recommendation_period_is_known(
        recommendations_df
    )
    and recommendation_start == context.filters.start_date
    and recommendation_end == context.filters.end_date
)

if not recommendation_matches_selection:
    st.warning(
        localized_text(
            language,
            (
                "Kayıtlı bütçe önerileri seçilen döneme ait değil. "
                "Tarih bölümünü açıp “Seçilen Dönemi Analiz Et” "
                "düğmesine basın. Doğru dönem üretilmeden eski bütçe "
                "önerileri gösterilmeyecektir."
            ),
            (
                "The saved budget recommendations do not belong to the "
                "selected period. Open the Date section and click "
                "“Analyze Selected Period”. Stale recommendations will "
                "not be shown until the correct period is generated."
            ),
        ),
        icon="⚠️",
    )
    render_read_only_footer(language)
    st.stop()

optimizer_data = prepare_optimizer_data(
    recommendations_df,
    portfolio_df,
)

st.info(
    localized_text(
        language,
        (
            "Bu sayfadaki öneriler seçilen tarih dönemi için "
            "Google Ads ve GA4 verileriyle üretilen pipeline "
            "çıktılarıdır."
        ),
        (
            "Recommendations on this page are pipeline outputs "
            "generated from Google Ads and GA4 data for the "
            "selected date period."
        ),
    )
)

if available_start is not None and available_end is not None:
    st.caption(
        localized_text(
            language,
            "Mevcut günlük veri dönemi: ",
            "Available daily data period: ",
        )
        + available_start.strftime("%d.%m.%Y")
        + " — "
        + available_end.strftime("%d.%m.%Y")
        + " | "
        + localized_text(
            language,
            "Son çıktı: ",
            "Latest output: ",
        )
        + get_latest_output_time()
    )


# ---------------------------------------------------------
# PLAN CONTROLS
# ---------------------------------------------------------

st.subheader(
    localized_text(
        language,
        "Bütçe Planı",
        "Budget Plan",
    )
)

plan_labels = {
    localized_text(
        language,
        "Model Önerisi",
        "Model Recommendation",
    ): "model",
    localized_text(
        language,
        "Portföy Dengeli",
        "Portfolio Balanced",
    ): "portfolio",
    localized_text(
        language,
        "Özel Toplam Bütçe",
        "Custom Total Budget",
    ): "custom",
}

control_columns = st.columns(
    [1.4, 1.0]
)

selected_plan_label = control_columns[0].selectbox(
    localized_text(
        language,
        "Plan Modu",
        "Plan Mode",
    ),
    options=list(plan_labels.keys()),
    key="budget_optimizer_plan_mode",
)

plan_mode = plan_labels[selected_plan_label]
current_total_budget = float(
    optimizer_data["CurrentSpendCanonical"].sum()
)

custom_total_budget: float | None = None

if plan_mode == "custom":
    custom_total_budget = control_columns[1].number_input(
        localized_text(
            language,
            "Toplam Günlük Bütçe (₺)",
            "Total Daily Budget (₺)",
        ),
        min_value=0.0,
        value=current_total_budget,
        step=max(current_total_budget * 0.05, 1.0),
        key="budget_optimizer_custom_total",
    )
else:
    mode_explanation = {
        "model": localized_text(
            language,
            (
                "Her kampanya için modelin en yüksek "
                "optimizasyon skoruyla seçtiği bütçe."
            ),
            (
                "The budget selected by the model's highest "
                "optimization score for each campaign."
            ),
        ),
        "portfolio": localized_text(
            language,
            (
                "Toplam mevcut bütçeyi koruyarak kampanyalar "
                "arasında optimize edilmiş dağılım."
            ),
            (
                "An optimized allocation across campaigns "
                "while preserving the current total budget."
            ),
        ),
    }

    control_columns[1].info(
        mode_explanation[plan_mode]
    )

budget_plan = build_budget_plan(
    optimizer_data=optimizer_data,
    plan_mode=plan_mode,
    custom_total_budget=custom_total_budget,
)

budget_plan = interpolate_campaign_predictions(
    plan=budget_plan,
    scenarios=scenarios_df,
)

outside_count = int(
    budget_plan["OutsideScenarioRange"].sum()
)

if outside_count > 0:
    st.warning(
        localized_text(
            language,
            (
                f"{outside_count} kampanyanın planlanan bütçesi "
                "eğitilmiş senaryo aralığının dışında. Tahmin "
                "en yakın güvenli senaryo sınırından hesaplandı."
            ),
            (
                f"{outside_count} campaign budgets are outside "
                "the trained scenario range. Predictions use "
                "the nearest safe scenario boundary."
            ),
        )
    )


# ---------------------------------------------------------
# PLAN RESULTS
# ---------------------------------------------------------

render_plan_kpis(
    budget_plan,
    language,
)

st.divider()
st.subheader(
    localized_text(
        language,
        "Mevcut ve Planlanan Günlük Bütçe Dağılımı",
        "Current and Planned Daily Budget Allocation",
    )
)

render_allocation_chart(
    budget_plan,
    language,
)

st.subheader(
    localized_text(
        language,
        "Kampanya Bütçe Planı",
        "Campaign Budget Plan",
    )
)

plan_table = build_plan_table(
    budget_plan,
    language,
)

st.dataframe(
    plan_table,
    width="stretch",
    hide_index=True,
    column_config=plan_table_column_config(
        language
    ),
    key="budget_optimizer_plan_table",
)


# ---------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------

st.divider()
st.subheader(
    localized_text(
        language,
        "Bu Öneri Neden Verildi?",
        "Why This Recommendation?",
    )
)

if (
    not recommendations_df.empty
    and "Campaign" in recommendations_df.columns
):
    explain_campaign = st.selectbox(
        localized_text(
            language,
            "Açıklanacak Kampanya",
            "Campaign to Explain",
        ),
        options=recommendations_df["Campaign"].astype(str).tolist(),
        key="budget_optimizer_explain_campaign",
    )

    explain_row = recommendations_df.loc[
        recommendations_df["Campaign"].astype(str) == str(explain_campaign)
    ].iloc[0]

    why_text = str(
        explain_row.get("WhyThisRecommendation", "") or ""
    ).strip()

    if why_text:
        st.info(why_text)
    else:
        st.caption(
            localized_text(
                language,
                "Bu çıktı için SHAP açıklaması bulunamadı.",
                "No SHAP explanation is available for this output.",
            )
        )

    driver_cols = st.columns(2)
    revenue_text = str(
        explain_row.get("RevenueTopDrivers", "") or ""
    ).strip()
    conversion_text = str(
        explain_row.get("ConversionTopDrivers", "") or ""
    ).strip()

    with driver_cols[0]:
        st.markdown(
            "**"
            + localized_text(
                language,
                "Gelir tahmini — en güçlü 3 model etkisi",
                "Revenue forecast — Top 3 model contributions",
            )
            + "**"
        )
        st.write(revenue_text or "—")

    with driver_cols[1]:
        st.markdown(
            "**"
            + localized_text(
                language,
                "Dönüşüm tahmini — en güçlü 3 model etkisi",
                "Conversion forecast — Top 3 model contributions",
            )
            + "**"
        )
        st.write(conversion_text or "—")

    st.caption(
        localized_text(
            language,
            (
                "SHAP değerleri model tahminine katkıyı açıklar; "
                "gerçek dünyada nedensellik iddiası değildir."
            ),
            (
                "SHAP values explain contribution to the model prediction; "
                "they do not establish real-world causality."
            ),
        )
    )


# ---------------------------------------------------------
# SCENARIO EXPLORER
# ---------------------------------------------------------

st.divider()
render_scenario_explorer(
    scenarios_df,
    language,
)


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

st.divider()
st.subheader(
    localized_text(
        language,
        "Bütçe Planını Dışa Aktar",
        "Export Budget Plan",
    )
)

render_export_buttons(
    csv_dataframe=plan_table,
    excel_sheets={
        "Budget Plan": plan_table,
        "Source Recommendations": recommendations_df,
        "Portfolio Allocation": portfolio_df,
        "Scenario Data": scenarios_df,
    },
    file_name="budget_optimizer_report",
    language=language,
    key_prefix="budget_optimizer",
)

render_read_only_footer(language)
