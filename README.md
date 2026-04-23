# Ads ML Budget Intelligence Pipeline

## Project Overview

This project is a comprehensive decision intelligence engine developed for a US-based e-commerce project. It is designed to transform digital marketing campaign data into actionable, forward-looking budget strategies. Moving beyond standard analytics and reporting dashboards, this is a robust, production-grade machine learning system that predicts future ad performance, simulates multiple budget allocation scenarios, and prescribes optimal capital distribution to maximize return on investment (ROI).

## Problem Statement

Traditional digital advertising reporting focuses heavily on descriptive analytics—reporting on past impressions, clicks, and Return on Ad Spend (ROAS) after the budget is already spent. Most marketing teams lack the capability to look forward and systematically evaluate how hypothetical budget adjustments will impact future outcomes. This reactive approach inevitably leads to wasted ad spend on underperforming campaigns, missed growth opportunities on high-potential ads, and suboptimal capital distribution across the entire marketing portfolio.

## Solution Approach

This project transitions the marketing workflow from a reactive, descriptive model to a predictive and prescriptive paradigm.

- **Predictive Forecasting:** By extracting historical performance data, the system engineers predictive features and trains machine learning models (Random Forest Regression) to forecast future conversions and revenue.
- **Prescriptive Optimization:** It simulates varying budget scenarios (e.g., 20% reduction, baseline, 20% increase) to evaluate potential outcomes.
- **Actionable Output:** An optimization engine evaluates these scenarios using a multi-metric weighted scoring system to recommend precise budget actions (Increase, Decrease, Maintain, Review/Pause) with attached statistical confidence scores.

## Data Pipeline & Tools

- **Data Ingestion:** Automated extraction and processing of a digital marketing dataset.
- **Feature Engineering:** Dynamic calculation of core KPIs (CTR, CPC, CPA, ROAS, Profit), along with temporal features such as rolling averages and lag metrics to capture trend momentum.
- **Machine Learning Models:** `scikit-learn` (Random Forest Regression) used for predicting future conversions and revenue with rigorous train/test validation.
- **Optimization Engine:** Custom algorithmic scoring system balancing predicted Revenue, Profit, and ROAS.
- **Languages & Frameworks:** Python, Pandas, Scikit-Learn, NumPy.

## Dashboard & Insights Business Impact

The implementation of this intelligence pipeline delivers significant, tangible business value:

- **Capital Efficiency:** Proactively identifies underperforming campaigns _before_ additional budget is burned, allowing for immediate intervention.
- **ROI Maximization:** Reallocates capital dynamically to campaigns demonstrating the highest predicted marginal return, optimizing overall portfolio yield.
- **Automated Decision Support:** Replaces gut-feeling adjustments and manual spreadsheet calculations with scalable, data-driven portfolio optimization.
- **Risk Mitigation:** Integrated guardrails and confidence scoring prevent erratic spending and flag high-variance predictions for manual review.

## Decision Logic

The pipeline follows a strict, mathematical flow to arrive at its recommendations:

1. **KPI & Feature Engineering:** Transform raw clicks/impressions into financial metrics and temporal trends.
2. **ML Prediction:** Forecast next-period conversions and revenue based on historical patterns.
3. **Scenario Simulation:** Generate hypothetical outcomes for decreased, maintained, and increased budget levels.
4. **Weighted Scoring:** Evaluate simulated outcomes using a composite score that weights ROAS (40%), Profit (40%), and Revenue (20%).
5. **Action Recommendation:** Assign discrete actions based on the highest-scoring scenario.
6. **Portfolio Allocation:** Intelligently redistribute the total marketing budget proportionally based on the optimized recommendations.
7. **Confidence Scoring:** Apply statistical guardrails; scenarios with low confidence are automatically flagged for manual review rather than automated budget shifts.

## 🚀 Key Achievements

- **Built an End-to-End ML Pipeline:** Engineered a fully functional decision intelligence system capable of processing and analyzing large-scale digital marketing datasets.
- **Developed Scenario Simulation:** Created a robust simulation engine capable of generating and evaluating multiple budget outcomes simultaneously to find the optimal path.
- **Implemented Automated Guardrails:** Designed an advanced confidence-scoring mechanism that safely automates budget recommendations while mitigating financial risk.
- **Standardized Budget Allocation:** Transformed the complex, manual task of marketing budget planning into a reproducible, production-ready Python application.
- **Bridged Data to Business Strategy:** Successfully abstracted mathematical complexity and machine learning models into clear, prescriptive business actions.

---

### How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your credentials and desired settings
```

3. Execute the pipeline:

```bash
python src/ads_ml_budget_intelligence.py
```
