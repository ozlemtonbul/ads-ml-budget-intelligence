# Vicco Ads ML Budget Intelligence Pipeline

## Overview
The Vicco Ads ML Budget Intelligence Pipeline is a comprehensive decision intelligence engine designed to transform Google Ads campaign data into actionable, forward-looking budget strategies. This is not a standard analytics script or a simple reporting dashboard; it is a robust, production-grade machine learning system that predicts performance, simulates budget scenarios, and prescribes optimal capital allocation to maximize return on investment.

## Problem Statement
Traditional Google Ads reporting focuses heavily on descriptive analytics—reporting on past impressions, clicks, and ROAS after the fact. Most companies lack the capability to look forward and systematically evaluate how budget adjustments will impact future outcomes. This reactive approach leads to wasted ad spend, missed opportunities, and suboptimal budget distribution across the portfolio.

## Solution
This project transitions the marketing workflow from a descriptive to a predictive and prescriptive paradigm. By integrating directly with the Google Ads API, the system automatically extracts data, engineers predictive features, and trains machine learning models to forecast future conversions and revenue. It then simulates multiple budget scenarios and employs an optimization engine to recommend precise budget actions (Increase, Decrease, Maintain, Pause) with attached confidence scores.

## Key Capabilities
- **API Integration:** Automated, secure data extraction from Google Ads.
- **KPI Engineering:** Dynamic calculation of core metrics (CTR, CPC, CPA, ROAS, Profit).
- **ML Forecasting:** Predictive modeling using Random Forest Regression for conversions and revenue.
- **Scenario Simulation:** Evaluation of reduced, unchanged, and increased budget impacts.
- **Optimization Logic:** Multi-metric weighted scoring system balancing revenue, profit, and ROAS.
- **Portfolio Allocation:** Intelligent, dynamic redistribution of total marketing budget.
- **Confidence Scoring:** Statistical confidence intervals and scores for every recommendation.
- **Guardrails:** Automated safety limits for low-confidence scenarios to prevent erratic spending.

## System Architecture
The pipeline follows a strict, systematic flow:
1. **Google Ads API** → 2. **Data Extraction** → 3. **KPI Engineering** → 4. **Feature Engineering** (Lags & Rolling Averages) → 5. **ML Prediction** (Random Forest) → 6. **Scenario Simulation** → 7. **Optimization Engine** → 8. **Recommendation System** → 9. **Portfolio Budget Allocation** → 10. **Confidence Scoring & Guardrails** → 11. **Final Outputs**

## Business Value
- **Reducing Wasted Ad Spend:** Identifies underperforming campaigns before budget is burned.
- **Improving ROI:** Reallocates capital to campaigns with the highest predicted marginal return.
- **Smarter Budget Allocation:** Replaces gut-feeling decisions with data-driven portfolio optimization.
- **Scalable Decision Automation:** Standardizes the complex task of budget planning into a reproducible, systematic engine.

## Why This Project Matters
This project is built as a true **Decision Intelligence System**. It abstracts away the complexity of machine learning and mathematical optimization to deliver clear, business-ready recommendations. It is engineered for production, ensuring reliability, scalability, and direct impact on the bottom line.

## Outputs
Upon execution, the system generates:
- Comprehensive scenario simulation results.
- Campaign-level action recommendations (Increase Budget, Decrease Budget, Maintain, Review/Pause).
- Optimized portfolio budget allocations.
- Feature importance analysis for the predictive models.
- Model performance metrics (MAE, RMSE, R²).

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Google Ads credentials and desired settings
```

3. Execute the pipeline:
```bash
python src/vicco_ads_ml_budget_intelligence.py
```

## Author
Ozlem Tonbul
