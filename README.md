# Ads Budget Intelligence

> **Enterprise AI-Powered Marketing Decision Intelligence Platform**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Google Ads API](https://img.shields.io/badge/Google_Ads_API-Integrated-success)
![GA4](https://img.shields.io/badge/GA4-Integrated-success)
![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supported-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Pytest](https://img.shields.io/badge/Tested_with-Pytest-success)
![Tests](https://github.com/ozlemtonbul/ads-ml-budget-intelligence/actions/workflows/tests.yml/badge.svg)
---

# Overview

Ads Budget Intelligence is an enterprise-ready Decision Intelligence platform that automates Google Ads budget optimization using predictive analytics, feature engineering, machine learning, business rule evaluation, and AI-generated executive commentary.

The platform integrates Google Ads, Google Analytics 4 (GA4), PostgreSQL, Docker, and Power BI into a single production-style analytics pipeline capable of supporting data-driven marketing decisions.

Unlike traditional reporting solutions, the platform predicts future campaign performance, evaluates multiple budget allocation scenarios, and recommends the most profitable investment strategy while maintaining configurable ROAS targets.

---

# Project Highlights

- Enterprise-ready modular architecture
- Google Ads API integration
- Google Analytics 4 integration
- Feature Engineering pipeline
- Random Forest prediction models
- Budget optimization engine
- Portfolio allocation engine
- AI-generated executive commentary
- PostgreSQL integration
- Docker support
- GitHub Actions CI
- **151 automated unit tests**
- Power BI ready

---

# Business Problem

Marketing teams frequently rely on historical reports and spreadsheets to manage advertising budgets.

This creates several operational challenges:

- Budget decisions are reactive instead of predictive.
- High-performing campaigns may not receive sufficient investment.
- Low-performing campaigns continue consuming budget.
- Seasonality and public holidays are often ignored.
- Decision making depends heavily on manual analysis.

As advertising portfolios grow, these limitations increase operational cost and reduce overall marketing efficiency.

---

# Business Value

The platform transforms campaign management into a predictive, automated decision-support process.

Key business outcomes include:

- Automated campaign monitoring
- Predictive revenue forecasting
- Budget optimization
- Portfolio-level budget allocation
- ROAS monitoring
- AI-assisted executive reporting
- Reduced manual reporting effort
- Improved decision consistency
- Enterprise-ready analytics architecture
---

# End-to-End Architecture

```text
                    Google Ads API
                           │
                           ▼
                 Google Ads Extractor
                           │
                           ▼
                  Google Analytics 4
                           │
                           ▼
                   GA4 Data Extractor
                           │
                           ▼
                 Feature Engineering
                           │
      ┌────────────────────┼────────────────────┐
      ▼                    ▼                    ▼
 Holiday Features     KPI Features        Lag Features
      │                    │                    │
      └────────────────────┼────────────────────┘
                           ▼
                 Training Dataset Builder
                           ▼
                Random Forest ML Models
          ┌────────────────┴────────────────┐
          ▼                                 ▼
Revenue Prediction               Conversion Prediction
          └────────────────┬────────────────┘
                           ▼
             Budget Scenario Simulation
                           ▼
              Budget Optimization Engine
                           ▼
               Recommendation Engine
      ┌──────────────┼──────────────┬──────────────┐
      ▼              ▼              ▼
CSV Reports     PostgreSQL      AI Commentary
                                      │
                                      ▼
                               Power BI Dashboard
```

---

# Enterprise Architecture

The project follows a modular enterprise architecture where each business capability is isolated into an independent Python module.

```text
config/
│
├── settings.py
├── google_ads_client.py
│
src/
│
├── extract/
│      Google Ads
│      GA4
│
├── features/
│      Feature Engineering
│      Reporting
│
├── models/
│      Machine Learning
│      Budget Optimization
│
├── recommendations/
│      Decision Engine
│
├── warehouse/
│      PostgreSQL
│
└── utils/
       Logger
```

---

# Technology Stack

| Layer | Technology |
|--------|------------|
| Programming | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| ML Algorithm | Random Forest Regression |
| APIs | Google Ads API |
| Analytics | Google Analytics Data API |
| AI | Anthropic Claude API |
| Database | PostgreSQL |
| Containerization | Docker |
| Reporting | Power BI |
| Testing | Pytest |
| CI | GitHub Actions |
| Version Control | Git + GitHub |

---

# Project Structure

```text
ads-budget-intelligence/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── config/
│
├── credentials/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── private/
│
├── docker/
│
├── docs/
│
├── notebooks/
│
├── outputs/
│   ├── csv/
│   ├── logs/
│   └── reports/
│
├── src/
│   ├── extract/
│   ├── features/
│   ├── models/
│   ├── recommendations/
│   ├── warehouse/
│   └── utils/
│
├── tests/
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt
```

---

# Data Flow

```text
Google Ads API
        │
        ▼
Google Ads Extractor
        │
        ▼
Google Analytics 4
        │
        ▼
GA4 Extractor
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ▼
Budget Optimization
        │
        ▼
Recommendation Engine
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
CSV   PostgreSQL   Executive Commentary
 │      │               │
 └──────┴───────────────┘
           ▼
     Power BI Dashboard
```

---

# Core Capabilities

The platform provides an end-to-end analytics workflow including:

- Automated Google Ads data extraction
- Google Analytics 4 integration
- KPI calculation
- Feature engineering
- Holiday intelligence
- Seasonality modelling
- Machine learning forecasting
- Budget scenario simulation
- Portfolio optimization
- Recommendation engine
- Confidence scoring
- Executive AI commentary
- CSV reporting
- PostgreSQL export
- Power BI integration
- Automated testing
- Continuous Integration (CI)
---

# Feature Engineering

The platform transforms raw advertising data into predictive features used by the machine learning models.

## KPI Features

The following business KPIs are calculated automatically:

- CTR (Click-Through Rate)
- CPC (Cost Per Click)
- CPA (Cost Per Acquisition)
- ROAS (Return on Ad Spend)
- Profit
- Conversion Rate

---

## Time Features

Time-based variables include:

- Day of Week
- Day of Month
- Month
- Quarter
- Weekend Flag

These features help the models learn weekly and seasonal behaviour patterns.

---

## Holiday Intelligence

The platform automatically enriches campaign data using the Turkish public holiday calendar.

Features include:

- Public Holiday Detection
- Holiday Name
- Pre-Holiday Detection (1–3 days before)
- Holiday ROAS Multiplier

This allows the prediction models to recognize demand spikes around important shopping periods.

---

## Seasonal Intelligence

Every observation is tagged with:

- Winter
- Spring
- Summer
- Autumn

Each season has configurable ROAS multipliers used during budget simulation.

---

## Lag Features

Historical campaign behaviour is captured using lag variables.

Generated features include:

- Previous Day Spend
- Previous Day Revenue
- Previous Day ROAS
- Previous Day Conversions
- 7-Day Rolling Average Spend
- 7-Day Rolling Average Revenue
- 7-Day Rolling Average ROAS

These variables improve the predictive capability of the machine learning models.

---

# Machine Learning Pipeline

The project trains two independent regression models.

## Model 1

Predicts:

- Next-period Revenue

Algorithm:

- Random Forest Regressor

---

## Model 2

Predicts:

- Next-period Conversions

Algorithm:

- Random Forest Regressor

---

## Model Validation

Every model is automatically evaluated using:

- MAE
- RMSE
- R² Score

Validation metrics are exported for monitoring and comparison.

---

# Budget Optimization Engine

After prediction, the platform simulates multiple investment strategies for every campaign.

Generated scenarios include:

- 50% Budget
- 75% Budget
- 100% Budget
- 120% Budget
- 150% Budget

Each scenario estimates:

- Expected Revenue
- Expected Profit
- Expected ROAS

The optimization engine automatically selects the highest-scoring scenario while respecting configurable business rules.

---

# Recommendation Engine

Recommendations are generated using both machine learning predictions and deterministic business rules.

Possible actions:

- Increase Budget
- Reduce Budget
- Maintain Budget
- Review Campaign

---

## Confidence Scoring

Each recommendation receives a confidence label:

- High
- Medium
- Low

Confidence is based on:

- Historical campaign data
- Model performance
- Prediction stability

---

## Portfolio Optimization

Campaigns are ranked according to:

- Expected Revenue
- Expected Profit
- ROAS
- Business Priority

This allows budget allocation decisions to be made at portfolio level instead of campaign level.

---

## Executive Commentary

The platform integrates Anthropic Claude to automatically generate business-friendly summaries.

Generated commentary includes:

- Campaign summary
- Portfolio summary
- Budget recommendations
- Executive insights

The objective is to translate technical model outputs into language understandable by marketing managers and executives.

---

# Output Files

The pipeline generates the following deliverables.

| Output | Description |
|---------|-------------|
| ads_daily_fact.csv | Daily campaign performance |
| ads_weekly_campaign_summary.csv | Weekly aggregation |
| ads_monthly_campaign_summary.csv | Monthly aggregation |
| ads_category_summary.csv | Category KPIs |
| ads_product_summary.csv | Product KPIs |
| ads_budget_scenarios.csv | Budget simulation |
| ads_budget_optimization_recommendations.csv | Final recommendations |
| ads_feature_importance.csv | Feature importance |
| ads_model_validation_metrics.csv | MAE, RMSE, R² |
| ads_holiday_impact.csv | Holiday analysis |
| ads_portfolio_budget_allocation.csv | Portfolio optimization |
| ads_recommendation_summary.csv | Executive recommendations |
| ads_portfolio_executive_commentary.txt | AI-generated commentary |

---

# Enterprise Reporting

The generated outputs can be consumed in two ways.

## CSV Reporting

All reports are exported as CSV files.

Suitable for:

- Excel
- Power BI
- Tableau
- Data validation

---

## PostgreSQL Reporting

The platform can automatically export every output table into PostgreSQL.

This enables enterprise reporting through:

- Power BI
- SQL
- Dashboards
Business Intelligence tools

---

# Performance Characteristics

The platform is designed to support enterprise-scale campaign analysis.

Capabilities include:

- Modular architecture
- Automated ETL
- Feature Engineering
- Predictive Analytics
- Scenario Simulation
- Rule-based Decision Engine
- AI Commentary
- PostgreSQL Reporting
- Power BI Integration
- Docker Deployment
- GitHub Actions CI

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ozlemtonbul/ads-ml-budget-intelligence.git

cd ads-ml-budget-intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

Copy the environment template.

```bash
cp .env.example .env
```

Configure the following services:

- Google Ads API
- Google Analytics 4
- Anthropic API
- PostgreSQL
- Target ROAS
- Date Range

---

# Running the Project

Run the complete pipeline:

```bash
python main.py
```

Expected output:

```text
Pipeline completed successfully.
```

---

# Docker

The project includes full Docker support.

Build and run:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up -d
```

Stop containers:

```bash
docker compose down
```

Docker provisions:

- Python Application
- PostgreSQL Database
- Shared Volumes
- Isolated Runtime Environment

This allows the complete analytics platform to run in an isolated environment.

---

# PostgreSQL

PostgreSQL support is optional and can be enabled through the environment configuration.

```env
POSTGRES_ENABLED=true
```

When enabled, the pipeline automatically exports all generated datasets into PostgreSQL.

Typical workflow:

```text
Google Ads + GA4
        │
        ▼
Python Pipeline
        ▼
PostgreSQL
        ▼
Power BI
```

---

# Power BI

The platform is designed for enterprise reporting.

Power BI can connect directly to PostgreSQL or consume generated CSV outputs.

Recommended dashboards:

- Executive Dashboard
- Campaign Performance Dashboard
- Portfolio Dashboard
- Recommendation Dashboard
- Holiday Performance Dashboard

---

# Automated Testing

The project contains a comprehensive automated testing suite.

Coverage includes:

- Google Ads Client
- Google Ads Extractor
- GA4 Extractor
- Feature Engineering
- KPI Calculations
- Budget Optimization
- Recommendation Engine
- Reporting
- PostgreSQL Export
- Logger
- Configuration
- Main Pipeline

Current status:

- ✅ 151 Automated Tests Passing
- ✅ Pytest
- ✅ Modular Unit Tests
- ✅ Integration Tests
- ✅ Mock API Testing

Run all tests:

```bash
python -m pytest -v
```

---

# Continuous Integration

GitHub Actions automatically:

- Installs dependencies
- Executes all automated tests
- Validates project integrity
- Reports build status

Every push to the main branch triggers the CI workflow automatically.

---

# Security

The following files are intentionally excluded from version control:

- `.env`
- `credentials/`
- `outputs/`
- `data/raw/`
- `data/private/`
- `__pycache__/`
- `.pytest_cache/`

No company credentials or confidential datasets are included in this repository.

---

# Roadmap

## Completed

- ✅ Google Ads API Integration
- ✅ Google Analytics 4 Integration
- ✅ Feature Engineering
- ✅ Machine Learning Models
- ✅ Budget Optimization Engine
- ✅ Recommendation Engine
- ✅ CSV Export
- ✅ PostgreSQL Export
- ✅ Docker Support
- ✅ Docker Compose
- ✅ GitHub Actions CI
- ✅ Automated Testing (151 Tests)
- ✅ Executive Commentary

## Planned

- Streamlit Dashboard
- SHAP Explainability
- Airflow Scheduling
- XGBoost Models
- LightGBM Models
- Real-Time Monitoring
- Power BI Executive Dashboard

---

# Current Status

## Project Status

Production-ready modular analytics platform with automated testing, Docker support, PostgreSQL integration and CI/CD workflow.
The current version provides:

- Enterprise-ready architecture
- Modular Python codebase
- Automated ETL pipeline
- Machine Learning forecasting
- Budget optimization
- AI-generated recommendations
- Docker support
- PostgreSQL integration
- GitHub Actions CI
- 151 passing automated tests

---

# Repository Statistics

| Category | Status |
|----------|--------|
| Python | ✅ |
| Google Ads API | ✅ |
| Google Analytics 4 | ✅ |
| Machine Learning | ✅ |
| Docker | ✅ |
| PostgreSQL | ✅ |
| GitHub Actions | ✅ |
| Automated Tests | ✅ 151 Passing |
| Power BI Ready | ✅ |
| Enterprise Architecture | ✅ |

---

# Future Enhancements

Planned future improvements include:

- Interactive Streamlit application
- SHAP model explainability
- Airflow orchestration
- Real-time monitoring
- Additional ML algorithms
- REST API deployment
- Kubernetes deployment
- Azure / AWS deployment
- Enterprise authentication

---

# Author

## Özlem Tonbul

AI-Powered Decision Intelligence • Business Intelligence • Marketing Analytics • Machine Learning • Data Analytics

Specializations:

- Marketing Intelligence
- Decision Intelligence
- Business Intelligence
- Machine Learning
- Google Ads Analytics
- Google Analytics 4
- E-commerce Analytics
- Operational Analytics

 Website: [ozlemtonbul.com](https://ozlemtonbul.com/)
 
GitHub: [ozlemtonbul](https://github.com/ozlemtonbul)

LinkedIn:[Özlem Tonbul](https://www.linkedin.com/in/ozlemtonbul/)

---


# License

This repository is provided for portfolio and educational purposes.

Company credentials, proprietary datasets, API keys and confidential business information are intentionally excluded from version control.

© 2026 Özlem Tonbul. All rights reserved.
