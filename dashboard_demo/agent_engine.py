from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

from dashboard_demo.utils import load_csv, load_text
from src.llm.manager import generate_text, get_llm_runtime_info


# ---------------------------------------------------------------------------
# Data loading and safe conversion
# ---------------------------------------------------------------------------

def dataframe_to_records(
    dataframe: pd.DataFrame,
    max_rows: int = 25,
) -> list[dict[str, Any]]:
    """Convert a dataframe into JSON-safe dictionaries."""
    if dataframe.empty:
        return []

    safe_df = dataframe.head(max_rows).copy()

    for column in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[column]):
            safe_df[column] = safe_df[column].astype(str)

    safe_df = safe_df.astype(object).where(
        pd.notna(safe_df),
        None,
    )

    return safe_df.to_dict(orient="records")


def load_agent_context() -> dict[str, Any]:
    """Load the latest analytics, optimization and model outputs."""
    daily_df = load_csv("ads_daily_fact.csv")
    recommendation_df = load_csv(
        "ads_budget_optimization_recommendations.csv"
    )
    portfolio_df = load_csv(
        "ads_portfolio_budget_allocation.csv"
    )
    summary_df = load_csv(
        "ads_recommendation_summary.csv"
    )
    metrics_df = load_csv(
        "ads_model_validation_metrics.csv"
    )
    feature_df = load_csv(
        "ads_feature_importance.csv"
    )
    commentary = load_text(
        "ads_portfolio_executive_commentary.txt"
    )

    return {
        "daily_performance": dataframe_to_records(
            daily_df.tail(30),
            max_rows=30,
        ),
        "recommendations": dataframe_to_records(
            recommendation_df,
            max_rows=100,
        ),
        "portfolio_allocation": dataframe_to_records(
            portfolio_df,
            max_rows=50,
        ),
        "recommendation_summary": dataframe_to_records(
            summary_df,
            max_rows=30,
        ),
        "model_validation": dataframe_to_records(
            metrics_df,
            max_rows=30,
        ),
        "feature_importance": dataframe_to_records(
            feature_df,
            max_rows=20,
        ),
        "executive_commentary": commentary,
    }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def get_first_value(
    row: dict[str, Any],
    candidates: list[str],
    default: Any = None,
) -> Any:
    """Return the first matching value while ignoring case and separators."""
    normalized_row = {
        str(key).lower().replace("_", "").replace(" ", ""): value
        for key, value in row.items()
    }

    for candidate in candidates:
        normalized_candidate = (
            candidate.lower()
            .replace("_", "")
            .replace(" ", "")
        )

        if normalized_candidate in normalized_row:
            return normalized_row[normalized_candidate]

    return default


def to_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to float safely."""
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_optional_number(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format numeric values safely."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def normalize_text(value: Any) -> str:
    """Normalize optional text values."""
    return str(value or "").strip().lower()


def contains_any(
    text: str,
    keywords: set[str],
) -> bool:
    """Return True when at least one keyword occurs in text."""
    return any(keyword in text for keyword in keywords)


# ---------------------------------------------------------------------------
# Language support
# ---------------------------------------------------------------------------

def detect_response_language(question: str) -> str:
    """Detect whether the answer should be Turkish or English."""
    turkish_characters = set("çğıöşüÇĞİÖŞÜ")

    turkish_keywords = {
        "neden",
        "hangi",
        "bütçe",
        "kampanya",
        "artır",
        "arttır",
        "azalt",
        "özet",
        "özetle",
        "performans",
        "gelir",
        "harcama",
        "öneri",
        "önerilen",
        "tahmini",
        "güven",
        "aksiyon",
        "daha fazla",
        "daha az",
        "en iyi",
        "en kötü",
        "risk",
        "riskli",
        "koru",
        "ölçekle",
        "kâr",
        "kar",
        "dönüşüm",
        "incele",
        "hedef",
        "altında",
        "üzerinde",
    }

    lowered_question = question.lower()

    if any(character in question for character in turkish_characters):
        return "Turkish"

    if contains_any(lowered_question, turkish_keywords):
        return "Turkish"

    return "English"


def get_welcome_message(language: str = "English") -> str:
    """Return the chat welcome message."""
    if language == "Turkish":
        return (
            "Merhaba. En güncel reklam performansını, bütçe önerilerini, "
            "ROAS risklerini, gelir ve kâr fırsatlarını, portföy dağılımını "
            "ve makine öğrenmesi sonuçlarını analiz edebilirim. "
            "Ne öğrenmek istersiniz?"
        )

    return (
        "Hello. I can analyze the latest advertising performance, budget "
        "recommendations, ROAS risks, revenue and profit opportunities, "
        "portfolio allocation and machine-learning results. "
        "What would you like to know?"
    )


# ---------------------------------------------------------------------------
# Recommendation classification
# ---------------------------------------------------------------------------

def classify_recommendation(
    row: dict[str, Any],
) -> str:
    """
    Classify a campaign.

    Explicit RecommendedAction has priority. BudgetChange is used only
    as a fallback when the action is missing or unrecognized.
    """
    action = normalize_text(
        get_first_value(row, ["RecommendedAction"])
    )
    predicted_roas = to_float(
        get_first_value(row, ["PredictedROAS", "ROAS"])
    )
    target_roas = to_float(
        get_first_value(row, ["TargetROAS"])
    )
    budget_change = to_float(
        get_first_value(row, ["BudgetChange"])
    )
    budget_change_pct = to_float(
        get_first_value(row, ["BudgetChangePct"])
    )

    if "review" in action:
        return "review"

    if "reduce" in action or "decrease" in action:
        return "decrease"

    if "maintain" in action or "keep" in action:
        return "maintain"

    if "increase" in action:
        if target_roas > 0 and predicted_roas >= target_roas:
            return "safe_increase"
        return "risky_increase"

    if budget_change < 0 or budget_change_pct < 0:
        return "decrease"

    if budget_change > 0 or budget_change_pct > 0:
        if target_roas > 0 and predicted_roas >= target_roas:
            return "safe_increase"
        return "risky_increase"

    if budget_change == 0 and budget_change_pct == 0:
        return "maintain"

    return "review"


def group_recommendations(
    recommendations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group and sort campaign recommendations."""
    grouped_rows: dict[str, list[dict[str, Any]]] = {
        "safe_increase": [],
        "risky_increase": [],
        "maintain": [],
        "decrease": [],
        "review": [],
    }

    for row in recommendations:
        category = classify_recommendation(row)
        grouped_rows[category].append(row)

    for category, rows in grouped_rows.items():
        if category == "decrease":
            rows.sort(
                key=lambda row: (
                    abs(
                        to_float(
                            get_first_value(
                                row,
                                ["BudgetChangePct"],
                            )
                        )
                    ),
                    to_float(
                        get_first_value(
                            row,
                            ["OptimizationScore"],
                        )
                    ),
                ),
                reverse=True,
            )
        else:
            rows.sort(
                key=lambda row: (
                    to_float(
                        get_first_value(
                            row,
                            ["BudgetChangePct"],
                        )
                    ),
                    to_float(
                        get_first_value(
                            row,
                            ["OptimizationScore"],
                        )
                    ),
                    to_float(
                        get_first_value(
                            row,
                            ["ProfitUplift"],
                        )
                    ),
                ),
                reverse=True,
            )

    return grouped_rows


# ---------------------------------------------------------------------------
# Priority, risk and explanation
# ---------------------------------------------------------------------------

def calculate_priority(
    row: dict[str, Any],
    category: str,
) -> str:
    """Calculate a deterministic business priority."""
    confidence = normalize_text(
        get_first_value(row, ["ConfidenceLevel"])
    )
    budget_change_pct = abs(
        to_float(
            get_first_value(row, ["BudgetChangePct"])
        )
    )
    revenue_uplift = to_float(
        get_first_value(row, ["RevenueUplift"])
    )
    profit_uplift = to_float(
        get_first_value(row, ["ProfitUplift"])
    )
    predicted_roas = to_float(
        get_first_value(row, ["PredictedROAS", "ROAS"])
    )
    target_roas = to_float(
        get_first_value(row, ["TargetROAS"])
    )

    score = 0

    if confidence == "high":
        score += 2
    elif confidence == "medium":
        score += 1

    if budget_change_pct >= 25:
        score += 2
    elif budget_change_pct >= 10:
        score += 1

    if revenue_uplift > 0:
        score += 1

    if profit_uplift > 0:
        score += 1

    if (
        category == "safe_increase"
        and target_roas > 0
        and predicted_roas >= target_roas
    ):
        score += 2

    if category == "risky_increase":
        score -= 1

    if category == "review":
        return "LOW"

    if score >= 6:
        return "HIGH"

    if score >= 3:
        return "MEDIUM"

    return "LOW"


def calculate_risk_score(
    row: dict[str, Any],
) -> tuple[int, list[str], list[str]]:
    """
    Calculate a transparent risk score from 0 to 100.

    Returns:
        score,
        Turkish risk reasons,
        English risk reasons
    """
    score = 0
    reasons_tr: list[str] = []
    reasons_en: list[str] = []

    predicted_roas = to_float(
        get_first_value(row, ["PredictedROAS", "ROAS"])
    )
    target_roas = to_float(
        get_first_value(row, ["TargetROAS"])
    )
    confidence = normalize_text(
        get_first_value(row, ["ConfidenceLevel"])
    )
    budget_change_pct = to_float(
        get_first_value(row, ["BudgetChangePct"])
    )
    budget_spike = get_first_value(
        row,
        ["BudgetSpike", "BudgetSpikeWarning"],
    )
    current_spend = to_float(
        get_first_value(row, ["CurrentSpend"])
    )
    action = normalize_text(
        get_first_value(row, ["RecommendedAction"])
    )
    revenue_uplift = to_float(
        get_first_value(row, ["RevenueUplift"])
    )
    profit_uplift = to_float(
        get_first_value(row, ["ProfitUplift"])
    )

    if target_roas > 0 and predicted_roas < target_roas:
        gap_pct = ((target_roas - predicted_roas) / target_roas) * 100
        score += min(35, int(gap_pct))
        reasons_tr.append("tahmini ROAS hedefin altında")
        reasons_en.append("predicted ROAS is below target")

    if budget_change_pct >= 50:
        score += 25
        reasons_tr.append("önerilen bütçe artışı %50 veya daha yüksek")
        reasons_en.append("recommended budget increase is 50% or higher")
    elif budget_change_pct >= 25:
        score += 15
        reasons_tr.append("önerilen bütçe artışı yüksek")
        reasons_en.append("recommended budget increase is high")

    if budget_spike:
        score += 15
        reasons_tr.append("bütçe sıçrama uyarısı var")
        reasons_en.append("a budget-spike warning is present")

    if confidence == "low":
        score += 20
        reasons_tr.append("model güven seviyesi düşük")
        reasons_en.append("model confidence is low")
    elif confidence == "medium":
        score += 10
        reasons_tr.append("model güven seviyesi orta")
        reasons_en.append("model confidence is medium")

    if current_spend <= 0:
        score += 15
        reasons_tr.append("aktif harcama yok")
        reasons_en.append("there is no active spend")

    if "risk" in action:
        score += 15
        reasons_tr.append("önerilen aksiyon açıkça ROAS riski içeriyor")
        reasons_en.append("the recommended action explicitly contains ROAS risk")

    if revenue_uplift < 0:
        score += 15
        reasons_tr.append("tahmini gelir artışı negatif")
        reasons_en.append("predicted revenue uplift is negative")

    if profit_uplift < 0:
        score += 15
        reasons_tr.append("tahmini kâr artışı negatif")
        reasons_en.append("predicted profit uplift is negative")

    return min(score, 100), reasons_tr, reasons_en


def build_decision_explanation(
    row: dict[str, Any],
    category: str,
    language: str,
) -> str:
    """Build a human-readable deterministic decision explanation."""
    predicted_roas = to_float(
        get_first_value(row, ["PredictedROAS", "ROAS"])
    )
    target_roas = to_float(
        get_first_value(row, ["TargetROAS"])
    )
    confidence = get_first_value(
        row,
        ["ConfidenceLevel"],
        "N/A",
    )
    budget_change_pct = to_float(
        get_first_value(row, ["BudgetChangePct"])
    )
    profit_uplift = to_float(
        get_first_value(row, ["ProfitUplift"])
    )

    if language == "Turkish":
        reasons: list[str] = []

        if target_roas > 0:
            if predicted_roas >= target_roas:
                reasons.append(
                    "tahmini ROAS hedef ROAS seviyesine eşit veya üzerindedir"
                )
            else:
                reasons.append(
                    "tahmini ROAS hedef ROAS seviyesinin altındadır"
                )

        reasons.append(f"güven seviyesi {confidence}")

        if budget_change_pct != 0:
            reasons.append(
                f"önerilen bütçe değişimi %{budget_change_pct:.2f}"
            )

        if profit_uplift > 0:
            reasons.append("tahmini kâr artışı pozitiftir")
        elif profit_uplift < 0:
            reasons.append("tahmini kâr artışı negatiftir")

        conclusions = {
            "safe_increase": (
                "Bu nedenle kontrollü şekilde ölçekleme uygundur."
            ),
            "risky_increase": (
                "Bu nedenle bütçe doğrudan yükseltilmemeli; kademeli test "
                "ve yakın performans takibi uygulanmalıdır."
            ),
            "maintain": (
                "Bu nedenle mevcut bütçeye yakın seviyenin korunması uygundur."
            ),
            "decrease": (
                "Bu nedenle bütçe azaltımı veya yeniden dağıtım "
                "değerlendirilmelidir."
            ),
            "review": (
                "Bu nedenle kampanya manuel olarak incelenmelidir."
            ),
        }

        return f"{'; '.join(reasons)}. {conclusions[category]}"

    reasons = []

    if target_roas > 0:
        if predicted_roas >= target_roas:
            reasons.append(
                "predicted ROAS meets or exceeds the target"
            )
        else:
            reasons.append(
                "predicted ROAS is below the target"
            )

    reasons.append(f"confidence is {confidence}")

    if budget_change_pct != 0:
        reasons.append(
            f"the recommended budget change is {budget_change_pct:.2f}%"
        )

    if profit_uplift > 0:
        reasons.append("predicted profit uplift is positive")
    elif profit_uplift < 0:
        reasons.append("predicted profit uplift is negative")

    conclusions = {
        "safe_increase": (
            "Controlled scaling is therefore appropriate."
        ),
        "risky_increase": (
            "The budget should not be increased immediately; use a staged "
            "test and close performance monitoring."
        ),
        "maintain": (
            "Keeping the budget near its current level is appropriate."
        ),
        "decrease": (
            "A budget reduction or reallocation should be considered."
        ),
        "review": (
            "The campaign should be reviewed manually."
        ),
    }

    return f"{'; '.join(reasons)}. {conclusions[category]}"


# ---------------------------------------------------------------------------
# Reusable campaign rendering
# ---------------------------------------------------------------------------

def append_campaign_details(
    lines: list[str],
    row: dict[str, Any],
    index: int,
    category: str,
    language: str,
    include_risk: bool = False,
) -> None:
    """Append complete campaign details to a response."""
    campaign = get_first_value(
        row,
        ["CampaignName", "Campaign"],
        "Unnamed campaign",
    )
    campaign_type = get_first_value(
        row,
        ["CampaignType"],
        "Unknown",
    )
    current_spend = get_first_value(
        row,
        ["CurrentSpend"],
    )
    recommended_budget = get_first_value(
        row,
        [
            "RecommendedBudget",
            "AllocatedBudget",
            "PortfolioBudget",
        ],
    )
    budget_change = get_first_value(
        row,
        ["BudgetChange"],
    )
    budget_change_pct = get_first_value(
        row,
        ["BudgetChangePct"],
    )
    predicted_roas = get_first_value(
        row,
        ["PredictedROAS", "ROAS"],
    )
    current_roas = get_first_value(
        row,
        ["ROAS"],
    )
    target_roas = get_first_value(
        row,
        ["TargetROAS"],
    )
    confidence = get_first_value(
        row,
        ["ConfidenceLevel"],
    )
    action = get_first_value(
        row,
        ["RecommendedAction"],
    )
    reason = get_first_value(
        row,
        ["RecommendationReason"],
    )
    roas_status = get_first_value(
        row,
        ["ROASStatus"],
    )
    revenue_uplift = get_first_value(
        row,
        ["RevenueUplift"],
    )
    revenue_uplift_pct = get_first_value(
        row,
        ["RevenueUpliftPct"],
    )
    profit_uplift = get_first_value(
        row,
        ["ProfitUplift"],
    )
    conversion_uplift = get_first_value(
        row,
        ["ConversionUplift"],
    )
    optimization_score = get_first_value(
        row,
        ["OptimizationScore"],
    )
    budget_spike_warning = get_first_value(
        row,
        ["BudgetSpikeWarning"],
    )

    priority = calculate_priority(
        row=row,
        category=category,
    )
    decision_explanation = build_decision_explanation(
        row=row,
        category=category,
        language=language,
    )
    risk_score, risk_reasons_tr, risk_reasons_en = (
        calculate_risk_score(row)
    )

    if language == "Turkish":
        lines.extend(
            [
                "",
                f"**{index}. {campaign}**",
                f"- Öncelik: {priority}",
                f"- Kampanya türü: {campaign_type}",
                (
                    "- Mevcut harcama: "
                    f"{format_optional_number(current_spend)}"
                ),
                (
                    "- Önerilen bütçe: "
                    f"{format_optional_number(recommended_budget)}"
                ),
                (
                    "- Bütçe değişimi: "
                    f"{format_optional_number(budget_change)} "
                    f"({format_optional_number(budget_change_pct)}%)"
                ),
                (
                    "- Mevcut ROAS: "
                    f"{format_optional_number(current_roas)}x"
                ),
                (
                    "- Tahmini ROAS: "
                    f"{format_optional_number(predicted_roas)}x"
                ),
                (
                    "- Hedef ROAS: "
                    f"{format_optional_number(target_roas)}x"
                ),
                f"- ROAS durumu: {roas_status or 'N/A'}",
                f"- Güven seviyesi: {confidence or 'N/A'}",
                (
                    "- Optimizasyon skoru: "
                    f"{format_optional_number(optimization_score)}"
                ),
                (
                    "- Tahmini gelir artışı: "
                    f"{format_optional_number(revenue_uplift)} "
                    f"({format_optional_number(revenue_uplift_pct)}%)"
                ),
                (
                    "- Tahmini kâr artışı: "
                    f"{format_optional_number(profit_uplift)}"
                ),
                (
                    "- Tahmini dönüşüm artışı: "
                    f"{format_optional_number(conversion_uplift)}"
                ),
                f"- Önerilen aksiyon: {action or 'N/A'}",
                f"- Model gerekçesi: {reason or 'N/A'}",
                f"- Karar açıklaması: {decision_explanation}",
            ]
        )

        if include_risk:
            risk_reason_text = (
                ", ".join(risk_reasons_tr)
                if risk_reasons_tr
                else "belirgin ek risk sinyali bulunmadı"
            )
            lines.extend(
                [
                    f"- Risk skoru: {risk_score}/100",
                    f"- Risk sinyalleri: {risk_reason_text}",
                ]
            )

        if budget_spike_warning:
            lines.append(
                f"- Bütçe artış uyarısı: {budget_spike_warning}"
            )

        return

    lines.extend(
        [
            "",
            f"**{index}. {campaign}**",
            f"- Priority: {priority}",
            f"- Campaign type: {campaign_type}",
            (
                "- Current spend: "
                f"{format_optional_number(current_spend)}"
            ),
            (
                "- Recommended budget: "
                f"{format_optional_number(recommended_budget)}"
            ),
            (
                "- Budget change: "
                f"{format_optional_number(budget_change)} "
                f"({format_optional_number(budget_change_pct)}%)"
            ),
            (
                "- Current ROAS: "
                f"{format_optional_number(current_roas)}x"
            ),
            (
                "- Predicted ROAS: "
                f"{format_optional_number(predicted_roas)}x"
            ),
            (
                "- Target ROAS: "
                f"{format_optional_number(target_roas)}x"
            ),
            f"- ROAS status: {roas_status or 'N/A'}",
            f"- Confidence level: {confidence or 'N/A'}",
            (
                "- Optimization score: "
                f"{format_optional_number(optimization_score)}"
            ),
            (
                "- Predicted revenue uplift: "
                f"{format_optional_number(revenue_uplift)} "
                f"({format_optional_number(revenue_uplift_pct)}%)"
            ),
            (
                "- Predicted profit uplift: "
                f"{format_optional_number(profit_uplift)}"
            ),
            (
                "- Predicted conversion uplift: "
                f"{format_optional_number(conversion_uplift)}"
            ),
            f"- Recommended action: {action or 'N/A'}",
            f"- Model reason: {reason or 'N/A'}",
            f"- Decision explanation: {decision_explanation}",
        ]
    )

    if include_risk:
        risk_reason_text = (
            ", ".join(risk_reasons_en)
            if risk_reasons_en
            else "no significant additional risk signal was found"
        )
        lines.extend(
            [
                f"- Risk score: {risk_score}/100",
                f"- Risk signals: {risk_reason_text}",
            ]
        )

    if budget_spike_warning:
        lines.append(
            f"- Budget increase warning: {budget_spike_warning}"
        )


def build_no_data_message(
    language: str,
) -> str:
    """Return a localized no-data message."""
    if language == "Turkish":
        return (
            "AI sağlayıcısı şu anda çevrimdışı ve kullanılabilir "
            "optimizasyon önerisi bulunmuyor. Son çıktıları oluşturmak "
            "için backend pipeline'ını çalıştırmalısın."
        )

    return (
        "The AI provider is currently offline and no optimization "
        "recommendation data is available. Run the backend pipeline "
        "to generate the latest campaign outputs."
    )


def build_offline_intro(
    language: str,
) -> list[str]:
    """Return the deterministic fallback introduction."""
    if language == "Turkish":
        return [
            (
                "LLM sağlayıcısı şu anda çevrimdışı. Bu cevap son "
                "optimizasyon çıktılarından doğrudan ve deterministik "
                "olarak üretildi."
            ),
            "",
        ]

    return [
        (
            "The LLM provider is currently offline, so this answer is "
            "generated deterministically from the latest optimization "
            "output."
        ),
        "",
    ]


# ---------------------------------------------------------------------------
# Executive and budget reports
# ---------------------------------------------------------------------------

def build_executive_summary_lines(
    grouped_rows: dict[str, list[dict[str, Any]]],
    language: str,
) -> list[str]:
    """Build an executive summary."""
    safe_count = len(grouped_rows["safe_increase"])
    risky_count = len(grouped_rows["risky_increase"])
    maintain_count = len(grouped_rows["maintain"])
    decrease_count = len(grouped_rows["decrease"])
    review_count = len(grouped_rows["review"])
    total_rows = sum(
        len(rows)
        for rows in grouped_rows.values()
    )

    total_revenue_uplift = sum(
        to_float(
            get_first_value(row, ["RevenueUplift"])
        )
        for rows in grouped_rows.values()
        for row in rows
    )
    total_profit_uplift = sum(
        to_float(
            get_first_value(row, ["ProfitUplift"])
        )
        for rows in grouped_rows.values()
        for row in rows
    )
    total_conversion_uplift = sum(
        to_float(
            get_first_value(row, ["ConversionUplift"])
        )
        for rows in grouped_rows.values()
        for row in rows
    )

    if language == "Turkish":
        lines = [
            "### Yönetici özeti",
            f"- Analiz edilen kampanya sayısı: {total_rows}",
            f"- Güvenli bütçe artışı: {safe_count}",
            f"- Riskli veya test gerektiren artış: {risky_count}",
            f"- Mevcut bütçeyi koru: {maintain_count}",
            f"- Bütçeyi azalt: {decrease_count}",
            f"- Manuel inceleme gereken kampanya: {review_count}",
            (
                "- Toplam tahmini gelir artışı: "
                f"{format_optional_number(total_revenue_uplift)}"
            ),
            (
                "- Toplam tahmini kâr artışı: "
                f"{format_optional_number(total_profit_uplift)}"
            ),
            (
                "- Toplam tahmini dönüşüm artışı: "
                f"{format_optional_number(total_conversion_uplift)}"
            ),
        ]
        return lines

    return [
        "### Executive summary",
        f"- Campaigns analyzed: {total_rows}",
        f"- Safe budget increases: {safe_count}",
        f"- Risky or test-required increases: {risky_count}",
        f"- Maintain current budget: {maintain_count}",
        f"- Decrease budget: {decrease_count}",
        f"- Campaigns requiring manual review: {review_count}",
        (
            "- Total predicted revenue uplift: "
            f"{format_optional_number(total_revenue_uplift)}"
        ),
        (
            "- Total predicted profit uplift: "
            f"{format_optional_number(total_profit_uplift)}"
        ),
        (
            "- Total predicted conversion uplift: "
            f"{format_optional_number(total_conversion_uplift)}"
        ),
    ]


def build_budget_analysis(
    recommendations: list[dict[str, Any]],
    language: str,
    max_rows_per_group: int = 5,
) -> str:
    """Build the full grouped budget analysis."""
    if not recommendations:
        return build_no_data_message(language)

    grouped_rows = group_recommendations(recommendations)
    lines = build_offline_intro(language)
    lines.extend(
        build_executive_summary_lines(
            grouped_rows,
            language,
        )
    )

    sections = [
        (
            "safe_increase",
            (
                "### Güvenli bütçe artışı önerileri"
                if language == "Turkish"
                else "### Safe budget increase recommendations"
            ),
        ),
        (
            "risky_increase",
            (
                "### Riskli veya kademeli test gerektiren artışlar"
                if language == "Turkish"
                else "### Risky increases requiring staged testing"
            ),
        ),
        (
            "maintain",
            (
                "### Mevcut bütçeyi koruma önerileri"
                if language == "Turkish"
                else "### Maintain current budget"
            ),
        ),
        (
            "decrease",
            (
                "### Bütçe azaltma veya yeniden dağıtım önerileri"
                if language == "Turkish"
                else "### Budget decrease or reallocation recommendations"
            ),
        ),
        (
            "review",
            (
                "### Manuel inceleme gereken kampanyalar"
                if language == "Turkish"
                else "### Campaigns requiring manual review"
            ),
        ),
    ]

    for category, title in sections:
        rows = grouped_rows[category]

        if not rows:
            continue

        lines.extend(["", title])

        for index, row in enumerate(
            rows[:max_rows_per_group],
            start=1,
        ):
            append_campaign_details(
                lines=lines,
                row=row,
                index=index,
                category=category,
                language=language,
                include_risk=(category == "risky_increase"),
            )

    if language == "Turkish":
        lines.extend(
            [
                "",
                "### Yönetim önerisi",
                (
                    "Güvenli artış grubundaki kampanyalar kontrollü şekilde "
                    "ölçeklenebilir. Riskli artış grubundaki kampanyalarda "
                    "önerilen bütçenin tamamı tek seferde uygulanmamalı; "
                    "kademeli test, günlük ROAS takibi ve ilk 7 günlük "
                    "performans kontrolü yapılmalıdır."
                ),
                "",
                "### Sınırlama",
                (
                    "Bu cevap deterministik fallback motoru tarafından "
                    "oluşturuldu. API anahtarı eklendiğinde LLM, bu doğrulanmış "
                    "analizi temel alarak daha ayrıntılı neden-sonuç analizi "
                    "ve yönetici yorumu üretecek."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Management recommendation",
                (
                    "Campaigns in the safe-increase group may be scaled in a "
                    "controlled manner. Risky increases should not be applied "
                    "in full immediately; use staged testing, daily ROAS "
                    "monitoring and a seven-day performance review."
                ),
                "",
                "### Limitation",
                (
                    "This response was generated by the deterministic fallback "
                    "engine. Once an API key is configured, the LLM will use "
                    "this validated analysis to add deeper cause-and-effect "
                    "analysis and managerial commentary."
                ),
            ]
        )

    return "\n".join(lines)


def build_executive_analysis(
    recommendations: list[dict[str, Any]],
    language: str,
) -> str:
    """Build a concise executive-only summary."""
    if not recommendations:
        return build_no_data_message(language)

    grouped_rows = group_recommendations(recommendations)
    lines = build_offline_intro(language)
    lines.extend(
        build_executive_summary_lines(
            grouped_rows,
            language,
        )
    )

    safe_rows = grouped_rows["safe_increase"][:3]
    risky_rows = grouped_rows["risky_increase"][:3]
    decrease_rows = grouped_rows["decrease"][:3]

    if language == "Turkish":
        lines.extend(["", "### En önemli yönetim aksiyonları"])
    else:
        lines.extend(["", "### Highest-priority management actions"])

    for title, rows, category in [
        (
            (
                "Güvenli ölçekleme"
                if language == "Turkish"
                else "Safe scaling"
            ),
            safe_rows,
            "safe_increase",
        ),
        (
            (
                "Riskli artış"
                if language == "Turkish"
                else "Risky increase"
            ),
            risky_rows,
            "risky_increase",
        ),
        (
            (
                "Azaltma veya yeniden dağıtım"
                if language == "Turkish"
                else "Decrease or reallocation"
            ),
            decrease_rows,
            "decrease",
        ),
    ]:
        for row in rows:
            campaign = get_first_value(
                row,
                ["CampaignName", "Campaign"],
                "Unnamed campaign",
            )
            action = get_first_value(
                row,
                ["RecommendedAction"],
                "N/A",
            )
            predicted_roas = get_first_value(
                row,
                ["PredictedROAS", "ROAS"],
            )
            target_roas = get_first_value(
                row,
                ["TargetROAS"],
            )
            priority = calculate_priority(row, category)

            if language == "Turkish":
                lines.append(
                    f"- **{title}:** {campaign} — Aksiyon: {action}; "
                    f"Tahmini ROAS: {format_optional_number(predicted_roas)}x; "
                    f"Hedef: {format_optional_number(target_roas)}x; "
                    f"Öncelik: {priority}"
                )
            else:
                lines.append(
                    f"- **{title}:** {campaign} — Action: {action}; "
                    f"Predicted ROAS: {format_optional_number(predicted_roas)}x; "
                    f"Target: {format_optional_number(target_roas)}x; "
                    f"Priority: {priority}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Specialized deterministic analyses
# ---------------------------------------------------------------------------

def build_risk_analysis(
    recommendations: list[dict[str, Any]],
    language: str,
    max_rows: int = 10,
) -> str:
    """Return only campaigns carrying meaningful risk signals."""
    if not recommendations:
        return build_no_data_message(language)

    risk_rows: list[tuple[int, dict[str, Any]]] = []

    for row in recommendations:
        risk_score, _, _ = calculate_risk_score(row)
        category = classify_recommendation(row)
        roas_status = normalize_text(
            get_first_value(row, ["ROASStatus"])
        )
        action = normalize_text(
            get_first_value(row, ["RecommendedAction"])
        )

        is_risky = (
            risk_score > 0
            and (
                category in {"risky_increase", "review"}
                or "below target" in roas_status
                or "risk" in action
                or get_first_value(
                    row,
                    ["BudgetSpike", "BudgetSpikeWarning"],
                )
                is not None
            )
        )

        if is_risky:
            risk_rows.append((risk_score, row))

    risk_rows.sort(
        key=lambda item: (
            item[0],
            abs(
                to_float(
                    get_first_value(
                        item[1],
                        ["BudgetChangePct"],
                    )
                )
            ),
        ),
        reverse=True,
    )

    lines = build_offline_intro(language)

    if language == "Turkish":
        lines.extend(
            [
                "### Risk analizi",
                f"- Risk sinyali taşıyan kampanya sayısı: {len(risk_rows)}",
                (
                    "- Risk skoru; ROAS hedef açığı, bütçe sıçraması, "
                    "model güveni, aktif harcama ve negatif iş etkisi "
                    "sinyallerinden hesaplanır."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "### Risk analysis",
                f"- Campaigns carrying risk signals: {len(risk_rows)}",
                (
                    "- Risk score is calculated from the ROAS target gap, "
                    "budget spikes, model confidence, active spend and "
                    "negative business-impact signals."
                ),
            ]
        )

    if not risk_rows:
        if language == "Turkish":
            lines.append(
                "Mevcut çıktılarda belirgin bir risk sinyali bulunamadı."
            )
        else:
            lines.append(
                "No significant risk signal was found in the current output."
            )
        return "\n".join(lines)

    for index, (risk_score, row) in enumerate(
        risk_rows[:max_rows],
        start=1,
    ):
        category = classify_recommendation(row)
        append_campaign_details(
            lines=lines,
            row=row,
            index=index,
            category=category,
            language=language,
            include_risk=True,
        )

    if language == "Turkish":
        lines.extend(
            [
                "",
                "### Risk yönetimi önerisi",
                (
                    "Yüksek risk skorlu kampanyalarda bütçe değişikliğini tek "
                    "seferde uygulamayın. Kademeli test, günlük ROAS kontrolü, "
                    "harcama limiti ve 7 günlük yeniden değerlendirme kullanın."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Risk-management recommendation",
                (
                    "Do not apply full budget changes immediately to high-risk "
                    "campaigns. Use staged testing, daily ROAS monitoring, "
                    "spend limits and a seven-day reassessment."
                ),
            ]
        )

    return "\n".join(lines)


def build_category_analysis(
    recommendations: list[dict[str, Any]],
    language: str,
    category: str,
    title_tr: str,
    title_en: str,
    max_rows: int = 10,
) -> str:
    """Build a report for one recommendation category."""
    if not recommendations:
        return build_no_data_message(language)

    grouped_rows = group_recommendations(recommendations)
    rows = grouped_rows.get(category, [])
    lines = build_offline_intro(language)
    lines.append(
        f"### {title_tr if language == 'Turkish' else title_en}"
    )

    if language == "Turkish":
        lines.append(f"- Kampanya sayısı: {len(rows)}")
    else:
        lines.append(f"- Campaign count: {len(rows)}")

    if not rows:
        if language == "Turkish":
            lines.append("Bu kategori için kampanya bulunamadı.")
        else:
            lines.append("No campaign was found for this category.")
        return "\n".join(lines)

    for index, row in enumerate(rows[:max_rows], start=1):
        append_campaign_details(
            lines=lines,
            row=row,
            index=index,
            category=category,
            language=language,
            include_risk=(category == "risky_increase"),
        )

    return "\n".join(lines)


def build_metric_ranking_analysis(
    recommendations: list[dict[str, Any]],
    language: str,
    metric_candidates: list[str],
    title_tr: str,
    title_en: str,
    metric_label_tr: str,
    metric_label_en: str,
    descending: bool = True,
    max_rows: int = 10,
    suffix: str = "",
) -> str:
    """Build a ranking report based on a selected metric."""
    if not recommendations:
        return build_no_data_message(language)

    rows = sorted(
        recommendations,
        key=lambda row: to_float(
            get_first_value(
                row,
                metric_candidates,
            )
        ),
        reverse=descending,
    )

    lines = build_offline_intro(language)
    lines.append(
        f"### {title_tr if language == 'Turkish' else title_en}"
    )

    for index, row in enumerate(rows[:max_rows], start=1):
        campaign = get_first_value(
            row,
            ["CampaignName", "Campaign"],
            "Unnamed campaign",
        )
        metric_value = get_first_value(
            row,
            metric_candidates,
        )
        predicted_roas = get_first_value(
            row,
            ["PredictedROAS", "ROAS"],
        )
        target_roas = get_first_value(
            row,
            ["TargetROAS"],
        )
        action = get_first_value(
            row,
            ["RecommendedAction"],
            "N/A",
        )
        confidence = get_first_value(
            row,
            ["ConfidenceLevel"],
            "N/A",
        )

        if language == "Turkish":
            lines.extend(
                [
                    "",
                    f"**{index}. {campaign}**",
                    (
                        f"- {metric_label_tr}: "
                        f"{format_optional_number(metric_value)}{suffix}"
                    ),
                    (
                        "- Tahmini ROAS: "
                        f"{format_optional_number(predicted_roas)}x"
                    ),
                    (
                        "- Hedef ROAS: "
                        f"{format_optional_number(target_roas)}x"
                    ),
                    f"- Önerilen aksiyon: {action}",
                    f"- Güven seviyesi: {confidence}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    f"**{index}. {campaign}**",
                    (
                        f"- {metric_label_en}: "
                        f"{format_optional_number(metric_value)}{suffix}"
                    ),
                    (
                        "- Predicted ROAS: "
                        f"{format_optional_number(predicted_roas)}x"
                    ),
                    (
                        "- Target ROAS: "
                        f"{format_optional_number(target_roas)}x"
                    ),
                    f"- Recommended action: {action}",
                    f"- Confidence level: {confidence}",
                ]
            )

    return "\n".join(lines)


def build_roas_analysis(
    recommendations: list[dict[str, Any]],
    language: str,
    show_worst: bool,
) -> str:
    """Build best- or worst-ROAS ranking."""
    if show_worst:
        return build_metric_ranking_analysis(
            recommendations=recommendations,
            language=language,
            metric_candidates=["PredictedROAS", "ROAS"],
            title_tr="En düşük ROAS değerine sahip kampanyalar",
            title_en="Campaigns with the lowest ROAS",
            metric_label_tr="Tahmini ROAS",
            metric_label_en="Predicted ROAS",
            descending=False,
            suffix="x",
        )

    return build_metric_ranking_analysis(
        recommendations=recommendations,
        language=language,
        metric_candidates=["PredictedROAS", "ROAS"],
        title_tr="En yüksek ROAS değerine sahip kampanyalar",
        title_en="Campaigns with the highest ROAS",
        metric_label_tr="Tahmini ROAS",
        metric_label_en="Predicted ROAS",
        descending=True,
        suffix="x",
    )


# ---------------------------------------------------------------------------
# Offline question router
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: dict[str, set[str]] = {
    "risk": {
        "risk",
        "riskli",
        "risky",
        "risk analysis",
        "risk analizi",
        "roas altında",
        "below target",
        "tehlikeli",
        "uyarı",
        "warning",
    },
    "decrease": {
        "azalt",
        "azaltılmalı",
        "bütçesi düşmeli",
        "reduce",
        "decrease",
        "cut budget",
        "less budget",
        "daha az bütçe",
        "yeniden dağıt",
        "reallocate",
    },
    "maintain": {
        "koru",
        "korunmalı",
        "sabit tut",
        "maintain",
        "keep budget",
        "hold budget",
    },
    "review": {
        "incele",
        "manuel inceleme",
        "review",
        "manual review",
        "harcama yok",
        "no spend",
    },
    "revenue": {
        "gelir",
        "revenue",
        "ciro",
        "revenue uplift",
        "gelir artışı",
    },
    "profit": {
        "kâr",
        "kar",
        "profit",
        "profit uplift",
        "kârlılık",
        "karlılık",
    },
    "conversion": {
        "dönüşüm",
        "conversion",
        "satın alma",
        "purchase",
    },
    "summary": {
        "özet",
        "özetle",
        "yönetici özeti",
        "executive summary",
        "summary",
        "genel durum",
        "overall",
    },
    "roas_worst": {
        "en kötü roas",
        "en düşük roas",
        "worst roas",
        "lowest roas",
        "düşük roas",
    },
    "roas_best": {
        "en iyi roas",
        "en yüksek roas",
        "best roas",
        "highest roas",
        "yüksek roas",
    },
    "budget": {
        "budget",
        "bütçe",
        "increase",
        "artır",
        "arttır",
        "more budget",
        "daha fazla bütçe",
        "recommended budget",
        "önerilen bütçe",
        "budget opportunity",
        "bütçe fırsatı",
        "scale",
        "ölçekle",
        "allocation",
        "dağılım",
    },
}


def detect_question_intent(
    question: str,
) -> str:
    """
    Detect deterministic analysis intent.

    Specific intents are checked before general budget intent.
    """
    question_lower = question.lower()

    intent_order = [
        "risk",
        "decrease",
        "maintain",
        "review",
        "revenue",
        "profit",
        "conversion",
        "roas_worst",
        "roas_best",
        "summary",
        "budget",
    ]

    for intent in intent_order:
        if contains_any(
            question_lower,
            INTENT_KEYWORDS[intent],
        ):
            return intent

    if "roas" in question_lower:
        return "roas_best"

    return "general"


def route_deterministic_question(
    question: str,
    context: dict[str, Any],
) -> str:
    """Route a user question to the appropriate deterministic analyst."""
    language = detect_response_language(question)
    intent = detect_question_intent(question)
    recommendations = context.get("recommendations", [])

    if intent == "risk":
        return build_risk_analysis(
            recommendations,
            language,
        )

    if intent == "decrease":
        return build_category_analysis(
            recommendations=recommendations,
            language=language,
            category="decrease",
            title_tr="Bütçesi azaltılacak veya yeniden dağıtılacak kampanyalar",
            title_en="Campaigns requiring budget reduction or reallocation",
        )

    if intent == "maintain":
        return build_category_analysis(
            recommendations=recommendations,
            language=language,
            category="maintain",
            title_tr="Mevcut bütçesi korunacak kampanyalar",
            title_en="Campaigns whose current budget should be maintained",
        )

    if intent == "review":
        return build_category_analysis(
            recommendations=recommendations,
            language=language,
            category="review",
            title_tr="Manuel inceleme gereken kampanyalar",
            title_en="Campaigns requiring manual review",
        )

    if intent == "revenue":
        return build_metric_ranking_analysis(
            recommendations=recommendations,
            language=language,
            metric_candidates=["RevenueUplift"],
            title_tr="En yüksek tahmini gelir artışı fırsatları",
            title_en="Highest predicted revenue-uplift opportunities",
            metric_label_tr="Tahmini gelir artışı",
            metric_label_en="Predicted revenue uplift",
        )

    if intent == "profit":
        return build_metric_ranking_analysis(
            recommendations=recommendations,
            language=language,
            metric_candidates=["ProfitUplift"],
            title_tr="En yüksek tahmini kâr artışı fırsatları",
            title_en="Highest predicted profit-uplift opportunities",
            metric_label_tr="Tahmini kâr artışı",
            metric_label_en="Predicted profit uplift",
        )

    if intent == "conversion":
        return build_metric_ranking_analysis(
            recommendations=recommendations,
            language=language,
            metric_candidates=["ConversionUplift"],
            title_tr="En yüksek tahmini dönüşüm artışı fırsatları",
            title_en="Highest predicted conversion-uplift opportunities",
            metric_label_tr="Tahmini dönüşüm artışı",
            metric_label_en="Predicted conversion uplift",
        )

    if intent == "roas_worst":
        return build_roas_analysis(
            recommendations,
            language,
            show_worst=True,
        )

    if intent == "roas_best":
        return build_roas_analysis(
            recommendations,
            language,
            show_worst=False,
        )

    if intent == "summary":
        return build_executive_analysis(
            recommendations,
            language,
        )

    if intent == "budget":
        return build_budget_analysis(
            recommendations,
            language,
        )

    commentary = str(
        context.get("executive_commentary", "")
    ).strip()

    if commentary and "skipped" not in commentary.lower():
        if language == "Turkish":
            return (
                "LLM sağlayıcısı şu anda çevrimdışı. Son oluşturulan "
                "yönetici değerlendirmesi aşağıdadır:\n\n"
                f"{commentary}"
            )
        return commentary

    if language == "Turkish":
        return (
            "AI sağlayıcısı şu anda çevrimdışı. Şu anda bütçe, risk, ROAS, "
            "gelir, kâr, dönüşüm, bütçe azaltma, bütçeyi koruma ve yönetici "
            "özeti sorularına deterministik cevap verebilirim."
        )

    return (
        "The AI provider is currently offline. I can currently provide "
        "deterministic answers about budgets, risk, ROAS, revenue, profit, "
        "conversions, budget reduction, budget maintenance and executive "
        "summaries."
    )


def build_general_fallback(
    question: str,
    context: dict[str, Any],
) -> str:
    """Backward-compatible wrapper for the deterministic router."""
    return route_deterministic_question(
        question=question,
        context=context,
    )


# ---------------------------------------------------------------------------
# Hybrid deterministic + LLM mode
# ---------------------------------------------------------------------------

def build_llm_prompt(
    question: str,
    history: list[dict[str, str]],
    context: dict[str, Any],
    deterministic_analysis: str,
    detected_intent: str,
) -> str:
    """Build the grounded hybrid prompt sent to the configured LLM."""
    response_language = detect_response_language(question)
    recent_history = history[-6:]

    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )

    context_text = json.dumps(
        context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return f"""
You are an enterprise advertising decision intelligence copilot.

Use only the supplied advertising data, machine-learning outputs and
validated deterministic analysis.

Detected intent: {detected_intent}
Reply language: {response_language}

Core operating model:
- The deterministic engine calculates, classifies and validates the result.
- Your role is to explain, summarize and prioritize the validated result.
- Answer the user's specific intent; do not provide unrelated report sections.
- Never contradict the deterministic analysis unless the supplied data
  contains a clear inconsistency. If it does, state the inconsistency.
- Never invent metrics, campaign names, causes or outcomes.
- Never claim that account changes were executed.
- Separate historical observations from model predictions.
- Distinguish CurrentSpend, RecommendedBudget and BudgetChange.
- Explain risk when PredictedROAS is below TargetROAS.
- Mention ConfidenceLevel and BudgetSpikeWarning when available.
- Preserve campaign names exactly as they appear in the data.
- Never expose credentials, tokens or API keys.
- Respond entirely in the requested reply language.
- Provide concise, prioritized and manager-ready recommendations.

Preferred response structure:
1. Direct answer
2. Evidence
3. Business impact
4. Recommended action
5. Confidence, risk or limitation

Conversation history:
{history_text}

Validated deterministic analysis:
{deterministic_analysis}

Analytics context:
{context_text}

Current question:
{question}
""".strip()


def ask_ads_agent(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """Main entry point used by the Streamlit Ask AI page."""
    clean_question = question.strip()

    if not clean_question:
        return "Lütfen bir soru girin. / Please enter a question."

    context = load_agent_context()
    runtime_info = get_llm_runtime_info()
    detected_intent = detect_question_intent(clean_question)

    deterministic_analysis = route_deterministic_question(
        question=clean_question,
        context=context,
    )

    if not runtime_info.get("ready", False):
        return deterministic_analysis

    prompt = build_llm_prompt(
        question=clean_question,
        history=history,
        context=context,
        deterministic_analysis=deterministic_analysis,
        detected_intent=detected_intent,
    )

    response = generate_text(
        prompt=prompt,
        max_tokens=1800,
        temperature=0.2,
    )

    if response:
        return response

    return deterministic_analysis



