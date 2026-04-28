# Ads ML Budget Intelligence Pipeline

## Project Overview

This project is a comprehensive decision intelligence engine developed for a US-based e-commerce project. It is designed to transform Google Ads campaign data into actionable, forward-looking budget strategies. Moving beyond standard analytics and reporting dashboards, this is a robust, production-grade machine learning system that predicts future ad performance, simulates multiple budget allocation scenarios, accounts for seasonal and holiday demand patterns, and prescribes optimal capital distribution to maximize return on investment (ROI).

The pipeline has been extended beyond its core ML engine with product and category-level granularity, calendar-aware seasonality modelling, ROAS target compliance tracking, and LLM-generated executive commentary — enabling both automated budget decisions and human-readable justification in a single run.

---

## Problem Statement

Traditional digital advertising reporting focuses heavily on descriptive analytics — reporting on past impressions, clicks, and Return on Ad Spend (ROAS) after the budget is already spent. Most marketing teams lack the capability to look forward and systematically evaluate how hypothetical budget adjustments will impact future outcomes. This reactive approach inevitably leads to wasted ad spend on underperforming campaigns, missed growth opportunities on high-potential ads, and suboptimal capital distribution across the entire marketing portfolio.

An additional layer of complexity arises from ignoring the structural patterns that affect performance: public holidays, religious observances, and seasonal demand shifts all significantly influence ROAS — yet most pipelines treat every day as equivalent.

---

## Solution Approach

This project transitions the marketing workflow from a reactive, descriptive model to a predictive and prescriptive paradigm.

- **Predictive Forecasting:** Historical performance data is extracted at campaign and ad group level, enriched with temporal, seasonal, and holiday features, and used to train Random Forest Regression models that forecast future conversions and revenue.
- **Prescriptive Optimization:** The system simulates varying budget scenarios (50%, 75%, 100%, 120%, 150% of current spend) and applies season and holiday multipliers to adjust predictions for known demand patterns.
- **ROAS Target Compliance:** Every budget recommendation is evaluated against a configurable ROAS target. Campaigns that meet the threshold receive standard increase recommendations; those that fall short are flagged with a risk warning.
- **Actionable Output:** A weighted scoring engine evaluates simulated outcomes across Revenue, Profit, and ROAS to recommend precise budget actions — Increase, Increase with ROAS Risk, Maintain, Reduce, or Pause/Review — with attached confidence scores.
- **Executive Commentary:** Anthropic Claude API generates plain-language, 3-sentence campaign summaries and a 5-sentence portfolio overview that non-technical stakeholders can act on directly.

---

## Data Pipeline & Tools

- **Data Ingestion:** Google Ads API — campaign and ad group level PURCHASE conversion data, extracted automatically for configurable date ranges.
- **Feature Engineering:** Dynamic calculation of core KPIs (CTR, CPC, CPA, ROAS, Profit), temporal features, rolling averages and lag metrics, holiday flags, pre-holiday eve detection, and seasonal ROAS multipliers.
- **Machine Learning Models:** `scikit-learn` Random Forest Regression for predicting future conversions and revenue, with rigorous train/test validation (MAE, RMSE, R²).
- **Optimization Engine:** Custom algorithmic scoring system balancing predicted Revenue (45%), Profit (35%), and ROAS (20%).
- **LLM Commentary:** Anthropic Claude API — campaign-level and portfolio-level plain-language summaries.
- **Languages & Frameworks:** Python, Pandas, NumPy, Scikit-Learn, Anthropic SDK, Google Ads API.

---

## Business Impact

- **Capital Efficiency:** Proactively identifies underperforming campaigns _before_ additional budget is burned, allowing for immediate intervention.
- **ROI Maximization:** Reallocates capital dynamically to campaigns demonstrating the highest predicted marginal return, optimizing overall portfolio yield.
- **Calendar-Aware Decisions:** Budget recommendations account for known demand spikes around public holidays, religious observances (Eid al-Fitr, Eid al-Adha), and seasonal shopping patterns — not just raw historical averages.
- **Automated Decision Support:** Replaces gut-feeling adjustments and manual spreadsheet calculations with scalable, data-driven portfolio optimization.
- **Non-Technical Accessibility:** LLM commentary translates model outputs into plain-language justifications that marketing managers and executives can read and act on directly.
- **Risk Mitigation:** Integrated ROAS target guardrails and confidence scoring prevent erratic spending and flag high-variance predictions for manual review.

---

## Decision Logic

The pipeline follows a strict, mathematical flow to arrive at its recommendations:

1. **Data Fetch** — Pull PURCHASE conversion data at campaign + ad group level from Google Ads API.
2. **KPI & Feature Engineering** — Compute financial metrics, time features, lag/rolling averages.
3. **Calendar Enrichment** — Tag each row with holiday flags, pre-holiday flags, and season multipliers.
4. **ML Training** — Train two Random Forest models: one for conversions, one for revenue.
5. **Scenario Simulation** — Generate hypothetical outcomes at 5 budget levels, adjusted for season/holiday.
6. **ROAS Target Check** — Evaluate each scenario against the configured TARGET_ROAS threshold.
7. **Weighted Scoring** — Score each scenario using Revenue (45%), Profit (35%), ROAS (20%).
8. **Action Recommendation** — Assign Increase / Maintain / Reduce / Pause per campaign, with ROAS gating.
9. **Portfolio Allocation** — Redistribute total budget proportionally based on optimized recommendations.
10. **Confidence Scoring** — Apply High/Medium/Low labels; Low confidence campaigns are flagged for manual review.
11. **LLM Commentary** — Generate plain-language summaries per campaign and for the full portfolio.

---

##  Key Achievements

- **Built an End-to-End ML Pipeline:** Engineered a fully functional decision intelligence system capable of processing and analyzing large-scale digital marketing datasets at campaign and product category level.
- **Developed Scenario Simulation:** Created a robust simulation engine capable of generating and evaluating multiple budget outcomes simultaneously, adjusted for real-world demand patterns including seasonality and public holidays.
- **Integrated Calendar Intelligence:** Modelled Turkish public holidays and religious observances (Eid al-Fitr, Eid al-Adha) and seasonal ROAS multipliers directly into the prediction and optimization layers.
- **Implemented ROAS-Gated Recommendations:** Designed a budget recommendation engine that gates increase decisions against a configurable ROAS target, separating high-confidence scaling opportunities from risk-flagged ones.
- **Implemented Automated Guardrails:** Designed an advanced confidence-scoring mechanism that safely automates budget recommendations while mitigating financial risk.
- **Added LLM-Powered Explainability:** Integrated Anthropic Claude API to generate plain-language executive summaries per campaign and at portfolio level, bridging the gap between model outputs and business decision-makers.
- **Standardized Budget Allocation:** Transformed the complex, manual task of marketing budget planning into a reproducible, production-ready Python application.
- **Bridged Data to Business Strategy:** Successfully abstracted mathematical complexity and machine learning models into clear, prescriptive business actions with human-readable justifications.

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

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables:**

```bash
cp .env.example .env
# Edit .env with your credentials and desired settings
```

**3. Execute the pipeline:**

```bash
python src/ads_ml_budget_intelligence.py
```

Output files will be written to the directory specified by `VICCO_OUTPUT_DIR` (default: `./output`).

---

## Author

Ozlem Tonbul
