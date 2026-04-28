# Ads ML Budget Intelligence Pipeline

## Executive Summary

This project is a machine learning–driven decision system designed to optimize Google Ads budget allocation using predictive analytics, scenario simulation, and business-focused optimization logic.

It transforms raw campaign data into actionable budget recommendations, enabling data-driven decision-making instead of manual, intuition-based planning.

Built for a multi-channel e-commerce environment, the system integrates forecasting, optimization, and explainability into a single pipeline.

---

## Business Problem

As the business scaled, revenue and ad spend increased significantly, but budget allocation decisions were still made manually based on historical performance.

Key challenges included:

* No predictive system for future performance
* Budget allocation based on past data rather than forecasts
* Lack of structured optimization across campaigns
* No integration of seasonal demand patterns and public holidays

Additionally, the Turkish e-commerce market is heavily influenced by seasonal events and holidays (such as Eid periods and back-to-school demand), which were not systematically considered in budget decisions.

---

## Solution

I designed and implemented a fully automated analytics pipeline that:

* Collects campaign-level data from Google Ads API
* Engineers performance features (CTR, CPC, CPA, ROAS, Profit)
* Builds predictive models for conversions and revenue
* Simulates multiple budget allocation scenarios
* Optimizes budget decisions based on business objectives
* Generates explainable insights using LLM-based commentary

---

## System Architecture

The pipeline consists of the following components:

### Data Layer

* Google Ads API integration (campaign & ad group level)
* Conversion-focused dataset (PURCHASE events)

### Feature Engineering

* KPI calculations: CTR, CPC, CPA, ROAS, Profit
* Time-based features: day, week, month, seasonality
* Lag features and rolling averages
* Holiday and pre-holiday flags
* Seasonal performance multipliers

### Machine Learning Models

* Random Forest Regression (2 models):

  * Conversion prediction
  * Revenue prediction
* Validation metrics:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * R² Score

### Scenario Simulation

* Simulates budget levels:

  * 50%, 75%, 100%, 120%, 150%
* Adjusts predictions using:

  * Seasonal effects
  * Holiday demand multipliers

### Optimization Engine

* Multi-objective scoring:

  * Revenue (45%)
  * Profit (35%)
  * ROAS (20%)
* Applies constraints:

  * ROAS threshold filtering
* Outputs:

  * Increase / Maintain / Decrease decisions

### Confidence Scoring

* Based on:

  * Data history depth
  * Model accuracy
* Labels:

  * High / Medium / Low confidence

### LLM Commentary

* Generates:

  * Campaign-level summaries
  * Portfolio-level insights
* Converts model outputs into business-readable insights

---

## Results & Business Impact

The system enabled:

* Transition from reactive to predictive budget management
* Improved decision quality through scenario-based optimization
* Better allocation of spend toward high-performing campaigns
* More efficient scaling of revenue-driving channels

Key observed outcomes:

* Revenue growth supported through optimized allocation
* Higher-value conversions prioritized
* Improved visibility into campaign performance drivers
* Structured decision-making replacing manual processes

---

## 📊 Dashboard & Insights

The project includes dashboard outputs to visualize:

* Campaign performance trends
* Budget vs revenue relationships
* ROAS distribution
* Scenario comparisons

These dashboards provide decision-makers with clear, visual insights to support budget optimization.

*(Dashboard screenshots should be included here in the repository)*

---

## Output Files

The system generates multiple structured outputs:

* `ads_budget_scenarios.csv` → Scenario simulations
* `ads_budget_optimization_recommendations.csv` → Final decisions
* `ads_portfolio_budget_allocation.csv` → Budget distribution
* `ads_model_validation_metrics.csv` → Model performance
* `ads_feature_importance.csv` → Key drivers
* `ads_category_summary.csv` → Category-level insights
* `ads_product_summary.csv` → Product-level performance
* `ads_holiday_impact.csv` → Holiday effects
* `ads_daily_fact.csv` → Daily metrics
* `ads_weekly_campaign_summary.csv` → Weekly trends
* `ads_monthly_campaign_summary.csv` → Monthly trends

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (Random Forest)
* Google Ads API
* Anthropic Claude API (LLM)
* Power BI (for dashboard visualization)

---

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
# Configure environment variables
python src/ads_ml_budget_intelligence.py
```

---

## Author

Ozlem Tonbul
Data-Driven Analytics & Decision Systems
