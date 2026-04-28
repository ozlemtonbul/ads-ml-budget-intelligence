# Ads ML Budget Intelligence Pipeline

## Overview

This is a machine learning pipeline I built independently to move Vicco's Google Ads budget management from manual, reactive decision-making to a data-driven, forward-looking system. The pipeline connects directly to the Google Ads API, engineers features from raw campaign data, trains predictive models, simulates budget scenarios, and outputs concrete budget recommendations — with seasonal and holiday context built in and plain-language executive summaries generated via LLM.

The system was built for a children's footwear brand operating across four sales channels (Web, Hepsiburada, Trendyol, LCW) and covers product categories including Okul Öncesi, Bebek, Çocuk, and İlk Adım.

---

## The Problem It Solved

The brand was scaling fast — revenue grew from ₺44.9M in 2023 to ₺72.7M in 2024, a 62% increase — but budget decisions were still being made manually, based on last month's numbers and intuition. With spend growing 65% year-over-year and four channels behaving differently, there was no systematic way to answer: *which campaigns should get more budget next week, and by how much?*

An additional challenge: the Turkish e-commerce calendar is heavily shaped by public holidays and religious observances — Eid al-Fitr, Eid al-Adha, back-to-school season — that create sharp, predictable demand spikes. These patterns were not being factored into budget decisions at all.

---

## What I Built

A fully automated pipeline that runs end-to-end from raw API data to actionable budget recommendations:

**Data layer:** Pulls PURCHASE conversion data at campaign and ad group level from Google Ads API across configurable date ranges. Product category and group are extracted from ad group naming conventions.

**Feature engineering:** Computes CTR, CPC, CPA, ROAS, and Profit. Adds time-based features (day of week, month, quarter, weekend flag), 1-day lag and 7-day rolling averages for spend, clicks, conversions, and revenue. Tags each row with Turkish public holiday flags, pre-holiday eve detection (1–3 days prior), and seasonal ROAS multipliers for Winter, Spring, Summer, and Autumn.

**Prediction:** Trains two Random Forest Regression models — one for next-period conversions, one for next-period revenue — with train/test validation reporting MAE, RMSE, and R².

**Scenario simulation:** Simulates five budget levels per campaign (50%, 75%, 100%, 120%, 150% of current spend). Predicted revenue is adjusted using a combined season and holiday multiplier, so a campaign running during Eid al-Adha in winter is not evaluated the same way as one running on a normal August Tuesday.

**Recommendation engine:** Scores each scenario using a weighted composite of predicted Revenue (45%), Profit (35%), and ROAS (20%). Budget increase decisions are gated against a configurable ROAS target — campaigns that would scale but fall below threshold are flagged separately as ROAS Risk rather than a clean Increase.

**Confidence scoring:** Each recommendation is labelled High, Medium, or Low confidence based on the campaign's data history depth and model R². Low confidence campaigns are automatically rerouted to Review rather than automated action.

**LLM commentary:** Anthropic Claude API generates a 3-sentence plain-language summary per campaign and a 5-sentence portfolio overview, translating model outputs into something a marketing manager or finance director can read and act on without needing to understand the model.

---

## Results

Over the period the pipeline was in use, Vicco's ad-attributed revenue continued to grow despite tighter efficiency constraints:

- Total revenue grew from ₺72.7M (2024) to ₺94.6M (2025) — **+30% year-over-year**
- Average basket value increased from ₺1,378 to ₺1,884 — **+37%** — while order volume declined, indicating the pipeline was successfully concentrating spend on higher-value conversions
- Ad spend grew 45% while revenue grew 30%, but this period included deliberate expansion into new channels (LCW) and the Trendyol marketplace scaling from ₺3.8M to ₺23.3M in the prior year
- ROAS held consistently in the 14–16x range across 2023–2024, declining modestly to ~14.6x in 2025 as new channel expansion temporarily compressed returns — an expected and planned outcome
- The pipeline flagged Hepsiburada and Trendyol as the highest-priority scaling targets based on predicted return curves; both saw 202% and 509% revenue growth respectively in the 2023–2024 period

---

## Technical Stack

| Component | Detail |
|---|---|
| Data source | Google Ads API — ad group level, PURCHASE conversions |
| ML models | scikit-learn Random Forest Regression (×2) |
| Feature set | 30+ features including KPIs, lag metrics, holiday flags, season multipliers |
| Optimization | Weighted scoring: Revenue 45%, Profit 35%, ROAS 20% |
| LLM | Anthropic Claude API |
| Language | Python — Pandas, NumPy, scikit-learn, anthropic, google-ads |

---

## Output Files

| File | Description |
|---|---|
| `ads_budget_scenarios.csv` | All simulated budget scenarios per campaign |
| `ads_budget_optimization_recommendations.csv` | Per-campaign optimization recommendations |
| `ads_portfolio_budget_allocation.csv` | Portfolio-normalized budget allocation |
| `ads_recommendation_summary.csv` | Final summary with LLM commentary column |
| `ads_model_validation_metrics.csv` | MAE, RMSE, R² for both models |
| `ads_feature_importance.csv` | Feature importances for both models |
| `ads_category_summary.csv` | KPI aggregation by product category and season |
| `ads_product_summary.csv` | KPI aggregation by product group and campaign |
| `ads_holiday_impact.csv` | Holiday vs pre-holiday vs normal day comparison |
| `ads_daily_fact.csv` | Daily KPIs with calendar context per campaign |
| `ads_weekly_campaign_summary.csv` | Weekly KPIs per campaign and category |
| `ads_monthly_campaign_summary.csv` | Monthly KPIs per campaign and category |
| `ads_portfolio_executive_commentary.txt` | LLM-generated portfolio summary (plain text) |
| `ads_rule_based_fallback_recommendations.csv` | Rule-based output when ML data is insufficient |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_ADS_DEVELOPER_TOKEN` | ✅ | — | Google Ads API developer token |
| `GOOGLE_ADS_CLIENT_ID` | ✅ | — | OAuth2 client ID |
| `GOOGLE_ADS_CLIENT_SECRET` | ✅ | — | OAuth2 client secret |
| `GOOGLE_ADS_REFRESH_TOKEN` | ✅ | — | OAuth2 refresh token |
| `GOOGLE_ADS_LOGIN_CUSTOMER_ID` | ✅ | — | MCC / manager account ID |
| `GOOGLE_ADS_CUSTOMER_ID` | ✅ | — | Target customer account ID |
| `DATE_MODE` | — | `last_60_days` | `yesterday` \| `last_30_days` \| `last_60_days` \| `custom` |
| `DATE_FROM` | custom only | — | Start date `YYYY-MM-DD` |
| `DATE_TO` | custom only | — | End date `YYYY-MM-DD` |
| `VICCO_OUTPUT_DIR` | — | `./output` | Output directory path |
| `TARGET_ROAS` | — | `3.0` | ROAS target threshold for compliance checks |
| `ANTHROPIC_API_KEY` | — | — | Anthropic API key for LLM commentary |
| `LLM_MAX_CAMPAIGNS` | — | `20` | Max campaigns to generate LLM commentary for |

---

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python src/ads_ml_budget_intelligence.py
```

---

## Author

Ozlem Tonbul
