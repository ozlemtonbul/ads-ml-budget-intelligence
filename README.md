## 📊 Live Interactive Dashboard
| | Link |
|--|--|
| Ads 2023/2024 · 2024/2025 · 2026 YTD | [View Dashboard →](https://ozlemtonbul.github.io/dashboards/ads_dashboard.html) |
| AI / ML / LLM Ads Budget Intelligence AI Agent Demo | [Launch Public Demo →](https://ads-ai-ml-llm-intelligence.streamlit.app/) |

# Ads Budget Intelligence AI Agent

> **Enterprise AI-Powered Marketing Decision Intelligence Platform**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Google Ads API](https://img.shields.io/badge/Google_Ads_API-Integrated-success)
![GA4](https://img.shields.io/badge/GA4-Integrated-success)
![Machine Learning](https://img.shields.io/badge/ML-RF%20%7C%20XGBoost%20%7C%20LightGBM-orange)
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
- Google Analytics 4 (GA4) integration
- Advanced Feature Engineering pipeline
- Multi-model Machine Learning Benchmarking (Random Forest, XGBoost, LightGBM)
- Automatic Best Model Selection
- SHAP Explainable AI (Top-3 Prediction Drivers)
- Evidence-Based Recommendation Engine
- Budget Optimization Engine
- Portfolio Budget Allocation Engine
- Provider-independent Multi-LLM architecture
- Anthropic Claude support
- OpenAI GPT support
- Google Gemini support
- AI-generated executive commentary
- Rule-based / deterministic AI fallback
- LLM usage guard and configurable cost controls
- Identifier Leakage Prevention
- Interactive Streamlit decision dashboard
- Public anonymized demo deployment
- PostgreSQL integration
- Docker support
- GitHub Actions CI
- **206 automated tests passing**
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
                Google Analytics 4 API
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
                           │
                           ▼
            Multi-Model Machine Learning
      ┌──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼
Random Forest     XGBoost       LightGBM
      └──────────────┼──────────────┘
                     ▼
         Model Benchmarking (MAE • RMSE • R²)
                     │
                     ▼
      Automatic Best Model Selection
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 Revenue Prediction     Conversion Prediction
         └───────────┬───────────┘
                     ▼
          SHAP Explainable AI
                     │
      Top-3 Prediction Drivers
                     │
                     ▼
       Budget Scenario Simulation
                     │
                     ▼
      Budget Optimization Engine
                     │
                     ▼
     Evidence-Based Recommendation Engine
                     │
                     ▼
               Multi-LLM Manager
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 Anthropic       OpenAI GPT     Google Gemini
   Claude
      └──────────────┼──────────────┘
                     ▼
 Executive Commentary (AI / Deterministic)
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 CSV Reports     PostgreSQL     Power BI
```
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
│      Google Ads Extraction
│      Google Analytics 4 Extraction
│
├── features/
│      Feature Engineering
│      Reporting
│
├── llm/
│      LLM Manager
│      LLM Usage Guard
│      Base Provider Interface
│
│      providers/
│      ├── Anthropic Claude Provider
│      ├── OpenAI GPT Provider
│      └── Google Gemini Provider
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

---

# Technology Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainable AI | SHAP |
| Machine Learning Algorithms | Random Forest, XGBoost, LightGBM |
| Feature Engineering | Custom Marketing Feature Pipeline |
| APIs | Google Ads API, Google Analytics Data API |
| AI | Multi-LLM (Anthropic Claude • OpenAI GPT • Google Gemini) |
| LLM Management | Provider Manager + Usage Guard |
| Database | PostgreSQL |
| Containerization | Docker |
| Interactive Dashboard | Streamlit |
| Visualization | Plotly |
| Reporting | CSV, PostgreSQL, Power BI |
| Testing | Pytest (206 Automated Tests) |
| CI/CD | GitHub Actions |
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
│   ├── llm/
│   │   ├── providers/
│   │   │   ├── anthropic_provider.py
│   │   │   ├── openai_provider.py
│   │   │   └── gemini_provider.py
│   │   ├── manager.py
│   │   ├── usage_guard.py
│   │   ├── base.py
│   │   └── __init__.py
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
        ▼
LLM Manager
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
Anthropic Claude   OpenAI GPT   Google Gemini
        │
        ▼
Executive Commentary
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
CSV Reports   PostgreSQL   Power BI Dashboard
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
- Configurable LLM request/token limits
- Interactive Streamlit dashboard
- Public anonymized demo mode
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

The platform benchmarks three independent machine learning regression algorithms using the same training and validation dataset.

## Machine Learning Algorithms

The following regression models are trained independently:

- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

Each algorithm predicts both:

- Next-period Revenue
- Next-period Conversions

---

## Automatic Model Benchmarking

Every model is automatically evaluated using identical train/test splits.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The benchmark results are exported to:

- ads_model_validation_metrics.csv

The platform automatically selects the highest-performing model independently for each prediction target.

Current benchmark:

- Revenue → XGBoost
- Conversions → XGBoost

This allows the production pipeline to use the most accurate validated model instead of relying on a fixed machine learning algorithm.

---

# Explainable AI (SHAP)

The platform provides transparent machine learning predictions using SHAP (SHapley Additive exPlanations).

For every campaign prediction the system automatically:

- Calculates SHAP values
- Identifies the Top-3 prediction drivers
- Explains why each prediction was generated
- Supplies evidence-based explanations to the recommendation engine
- Supports both deterministic and LLM-generated executive commentary

Generated output:

- ads_shap_explanations.csv

Campaign identifiers are intentionally excluded from predictive model training to prevent identifier leakage while remaining available for grouping, lag feature generation and reporting.

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

Recommendations are generated using:

- Machine Learning Predictions
- SHAP Explainability
- Deterministic Business Rules

Possible actions include:

- Increase Budget
- Reduce Budget
- Maintain Budget
- Review Campaign

Every recommendation is supported by explainable prediction evidence rather than model output alone.

---

## Confidence Scoring

Each recommendation receives a confidence label:

- High
- Medium
- Low

Confidence is calculated using:

- Historical campaign behaviour
- Prediction stability
- Validation performance
- Selected model confidence

---

## Portfolio Optimization

Campaigns are ranked according to:

- Expected Revenue
- Expected Profit
- ROAS
- Business Priority

Portfolio optimization enables budget allocation decisions across all campaigns instead of evaluating campaigns individually.

---

## Executive Commentary

The platform supports provider-independent AI-generated executive commentary through a centralized Multi-LLM Manager.

Supported providers:

- Anthropic Claude
- OpenAI GPT
- Google Gemini

The Recommendation Engine generates structured business prompts using:

- Machine Learning Predictions
- Budget Optimization Results
- Portfolio Analysis
- SHAP Explainability

The Multi-LLM Manager routes requests to the selected provider defined by the project configuration.

Generated commentary includes:

- Campaign Summary
- Portfolio Summary
- Budget Recommendations
- Executive Insights
- Business Risk Assessment
- Optimization Opportunities

Only one LLM provider is active at runtime.

If no API key is configured, or live generation is disabled, the platform automatically falls back to deterministic executive commentary without interrupting recommendation generation.

---

# LLM Cost & Usage Controls

Live LLM generation is optional and disabled by default.

Analytics, machine learning, optimization and deterministic recommendations remain fully operational without an LLM API key.

The project includes a dedicated `src/llm/usage_guard.py` layer to control API usage.

Current safeguards include:

- `LLM_ENABLED=false` by default
- Automatic provider selection (`LLM_PROVIDER=auto`)
- Configurable maximum response tokens
- Daily request limits
- Local usage tracking
- Automatic request blocking when limits are exceeded
- Safe deterministic fallback

API keys are never stored in source control.

---

# Interactive Streamlit Dashboard

The project includes an interactive Streamlit decision-support dashboard.

Dashboard modules include:

- Executive Overview
- Campaign Analysis
- Budget Optimizer
- AI Insights
- Ask AI
- Turkish / English interface
- Date-range comparison
- Campaign filtering
- Category filtering
- Channel filtering
- SHAP Explainability
- Model Benchmark Results

## Public Demo

A public portfolio demo is deployed with an anonymized dataset:

**Public Streamlit Demo:** https://ads-ai-ml-llm-intelligence.streamlit.app/

The public demo is intentionally isolated from the live production/local pipeline. It uses anonymized demo outputs and does **not** call live Google Ads or GA4 APIs. Live credentials, private company data, and local production outputs are not included in the public demo repository assets.

---

# Output Files

The pipeline generates the following deliverables.

| Output | Description |
|---------|-------------|
| ads_daily_fact.csv | Daily campaign performance dataset |
| ads_weekly_campaign_summary.csv | Weekly campaign performance summary |
| ads_monthly_campaign_summary.csv | Monthly campaign performance summary |
| ads_category_summary.csv | Category-level KPI summary |
| ads_product_summary.csv | Product-level KPI summary |
| ads_budget_scenarios.csv | Multi-scenario budget simulation results |
| ads_budget_optimization_recommendations.csv | Final campaign budget recommendations |
| ads_feature_importance.csv | Global machine learning feature importance |
| ads_model_validation_metrics.csv | Multi-model benchmark results (MAE, RMSE, R²) |
| ads_shap_explanations.csv | SHAP explainability results and Top-3 prediction drivers |
| ads_holiday_impact.csv | Holiday performance analysis |
| ads_portfolio_budget_allocation.csv | Portfolio-level budget allocation |
| ads_recommendation_summary.csv | Executive recommendation summary |
| ads_portfolio_executive_commentary.txt | AI-generated executive portfolio commentary |
| ga4_campaign_performance.csv | Google Analytics 4 campaign performance dataset |
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
- Business Intelligence tools
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
- Provider-independent Multi-LLM Architecture
- AI-generated Executive Commentary
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

- Google Ads API credentials
- Google Analytics 4 credentials
- LLM Provider (Anthropic Claude, OpenAI GPT, or Google Gemini)
- PostgreSQL connection
- Target ROAS
- Date Range

## Multi-LLM Configuration

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6

ANTHROPIC_API_KEY=

OPENAI_API_KEY=

GEMINI_API_KEY=
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
- Multi-model Machine Learning Benchmarking
- Automatic Best Model Selection
- Random Forest validation
- XGBoost validation
- LightGBM validation
- SHAP Explainability
- Top-3 Prediction Driver generation
- Identifier Leakage Prevention
- Budget Optimization
- Recommendation Engine
- Evidence-Based Recommendation Explanations
- Reporting
- PostgreSQL Export
- Logger
- Configuration
- Multi-LLM runtime management
- LLM daily usage guard / cost-control behavior
- Deterministic fallback behavior
- Main Pipeline

## Current Status

- ✅ 206 Automated Tests Passing
- ✅ Pytest
- ✅ Modular Unit Tests
- ✅ Integration Tests
- ✅ Mock API Testing
- ✅ Multi-model Benchmark Validation
- ✅ Automatic Model Selection Tests
- ✅ SHAP Explainability Tests
- ✅ Identifier Leakage Prevention Tests
- ✅ Provider-independent Multi-LLM Architecture
- ✅ LLM Manager
- ✅ Anthropic Claude Provider
- ✅ OpenAI GPT Provider
- ✅ Google Gemini Provider
- ✅ Rule-based / Deterministic Fallback
- ✅ LLM Daily Usage Guard
- ✅ LLM Cost-Control Tests

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

No company credentials or confidential datasets are included in this repository. The public Streamlit deployment uses anonymized demo data and is separated from live Google Ads, GA4, local outputs, and private credentials.

---

# Roadmap

## Completed

- ✅ Google Ads API Integration
- ✅ Google Analytics 4 Integration
- ✅ Advanced Feature Engineering
- ✅ Multi-Model Machine Learning Benchmarking
- ✅ Random Forest Regression
- ✅ XGBoost Regression
- ✅ LightGBM Regression
- ✅ Automatic Best Model Selection
- ✅ SHAP Explainable AI
- ✅ Top-3 Prediction Drivers
- ✅ Identifier Leakage Prevention
- ✅ Budget Optimization Engine
- ✅ Recommendation Engine
- ✅ Evidence-Based Recommendation Engine
- ✅ Multi-LLM Architecture
- ✅ LLM Manager
- ✅ Anthropic Provider
- ✅ OpenAI Provider
- ✅ Google Gemini Provider
- ✅ Rule-based / Deterministic Fallback
- ✅ LLM Daily Usage Guard
- ✅ LLM Cost Controls
- ✅ AI-Generated Executive Commentary
- ✅ CSV Export
- ✅ PostgreSQL Export
- ✅ Docker Support
- ✅ Docker Compose
- ✅ GitHub Actions CI
- ✅ Automated Testing (206 Tests)
- ✅ Interactive Streamlit Dashboard
- ✅ Public Anonymized Streamlit Demo

## Planned

- Airflow Scheduling
- Real-Time Monitoring
- Power BI Executive Dashboard
- Online Model Retraining
- Model Registry
- REST API Deployment
- Kubernetes Deployment
- Azure / AWS Production Deployment

---

# Current Status

## Project Status

Production-ready AI-powered marketing decision intelligence platform with automated testing, Docker support, PostgreSQL integration, CI/CD workflow, multi-model machine learning, and explainable AI.

The current version provides:

- Enterprise-ready modular architecture
- Automated ETL pipeline
- Google Ads + Google Analytics 4 integration
- Multi-model machine learning benchmarking
- Automatic best-model selection
- SHAP Explainable AI
- Evidence-based recommendation engine
- Budget optimization
- Portfolio optimization
- Provider-independent Multi-LLM architecture
- AI-generated executive commentary
- Interactive Streamlit decision dashboard
- Public anonymized demo deployment
- LLM usage guard and configurable API cost controls
- Docker support
- PostgreSQL integration
- GitHub Actions CI
- Power BI-ready outputs
- 206 passing automated tests

---

# Repository Statistics

| Category | Status |
|----------|--------|
| Python | ✅ |
| Google Ads API | ✅ |
| Google Analytics 4 | ✅ |
| Multi-Model Machine Learning | ✅ |
| Random Forest | ✅ |
| XGBoost | ✅ |
| LightGBM | ✅ |
| SHAP Explainable AI | ✅ |
| Automatic Model Selection | ✅ |
| Recommendation Engine | ✅ |
| Multi-LLM Support | ✅ |
| LLM Usage Guard | ✅ |
| Streamlit Dashboard | ✅ |
| Public Anonymized Demo | ✅ |
| Provider-Independent Architecture | ✅ |
| Docker | ✅ |
| PostgreSQL | ✅ |
| GitHub Actions | ✅ |
| Automated Tests | ✅ 206 Passing |
| Power BI Ready | ✅ |
| Enterprise Architecture | ✅ |

---

# Future Enhancements

Planned future improvements include:

- Airflow orchestration
- Real-time monitoring
- Online model retraining
- Model registry
- REST API deployment
- Kubernetes deployment
- Azure / AWS production deployment
- Enterprise authentication
- Advanced forecasting models
- MLOps pipeline

---

# Author

## Özlem Tonbul

**AI-Powered Decision Intelligence • Business Intelligence • Marketing Analytics • Machine Learning • Data Analytics**

### Specializations

- Marketing Intelligence
- Decision Intelligence
- Business Intelligence
- Machine Learning
- Explainable AI
- Google Ads Analytics
- Google Analytics 4
- E-commerce Analytics
- Operational Analytics

🌐 Website: https://ozlemtonbul.com

💻 GitHub: https://github.com/ozlemtonbul

💼 LinkedIn: https://www.linkedin.com/in/ozlemtonbul/

---

# License

This repository is provided for portfolio and educational purposes.

Company credentials, proprietary datasets, API keys, confidential business information and production datasets are intentionally excluded from version control.

© 2026 Özlem Tonbul. All rights reserved.