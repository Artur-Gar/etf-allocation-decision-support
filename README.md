# ETF Allocation Decision Support

## General Overview

This repository presents a business intelligence workflow for international equity ETF selection and allocation. The project addresses a portfolio construction problem in which funds must be screened not only by historical performance, but also by cost, liquidity, geographic exposure, industry composition, and macroeconomic context.

## Technical Overview

The implementation combines Python-based data engineering, Tableau dashboards, and a relational analytical model to transform raw ETF, market, and macro data into a decision-support story. The pipeline integrates iShares fund characteristics, Yahoo Finance price histories, and FRED macroeconomic indicators, then applies financial metric engineering and constrained Markowitz optimization to produce a final portfolio recommendation.

The optimization component should therefore be read as a structured allocation aid rather than a high-confidence forecasting engine. Given the well-known limits of mean-variance methods in predicting future market behavior, the main contribution of the project lies in building coherent dashboards and an analytically transparent workflow for ETF screening, comparison, and portfolio discussion.

Full report: [docs/ETF_allocation_support_report.pdf](docs/ETF_allocation_support_report.pdf)

## Key Techniques

- Multi-source data collection from iShares, Yahoo Finance, and FRED
- Monthly return engineering, cumulative index construction, momentum, and drawdown analysis
- ETF taxonomy standardization with optional OpenAI-assisted classification refinement
- Tableau-ready galaxy-schema modeling for ETF, geography, industry, and macro data
- Risk analytics based on annualized return, volatility, Sharpe ratio, correlation, and drawdown
- Markowitz-style portfolio optimization with long-only and max-weight constraints

## Results

The screening dashboard narrows the ETF universe to seven developed-market dividend funds with a median expense ratio of 0.4%, total AUM of $60.9bn, and near-zero median bid-ask spread. Within this filtered set, `IVVW` achieves the strongest standalone risk-adjusted performance with a Sharpe ratio of `1.18`.

The best-performing portfolio is the maximum-Sharpe allocation generated under a `3.0%` risk-free rate and a `30%` per-ETF cap. It reaches an expected return of `18.6%`, expected volatility of `8.4%`, Sharpe ratio of `1.848`, and maximum drawdown of `-4.0%`, with the largest weights assigned to `IVVW`, `IDV`, and `DVYA`.

### ETF Screener

The ETF Screener dashboard defines the initial investable universe by combining structural filters with practical fund-selection criteria. It allows the analyst to narrow the set of candidates by development group, style, size, sector, and management type while checking whether cost, liquidity, and scale remain acceptable. In the illustrated scenario, the screen retains seven developed-market dividend ETFs for deeper analysis.

<img src="docs/figures/Dashboard 1 Screener.png" alt="ETF Screener" width="900">

### Risk and Diversification

The Risk and Diversification dashboard compares the shortlisted ETFs through a portfolio-construction lens rather than a simple return ranking. It brings together annualized return, volatility, Sharpe ratio, correlation, and drawdown to identify which funds offer strong standalone performance and which contribute genuine diversification benefits. This step shows that `IVVW` dominates on risk-adjusted return while correlations across the remaining funds remain materially positive.

<img src="docs/figures/Dashboard 4 Risk.png" alt="Risk and Diversification" width="900">

### Exposure Analysis

The Exposure Analysis dashboard addresses the question of what the investor is actually buying once a fund passes the initial screen. It decomposes an ETF into country, regional, and industry weights so that apparent style labels can be validated against underlying holdings. In the example shown, `IDV` reveals a concentrated developed-market profile with substantial exposure to Europe and to the financials sector.

<img src="docs/figures/Dashboard 2 Exposure.png" alt="Exposure Analysis" width="900">

### Macro Context

The Macro Context dashboard adds an explanatory economic layer to the ETF selection process by linking fund exposure to country-level indicators. Instead of directly determining portfolio weights, it helps interpret whether the countries most represented in a selected ETF face stable or changing inflation, rate, or labor-market conditions. The example focuses on inflation dynamics across the main countries underlying `IDV`.

<img src="docs/figures/Dashboard 3 Macro.png" alt="Macro Context" width="900">

### Portfolio Recommendation

The Portfolio Recommendation dashboard translates the previous screening and diagnostic steps into a final allocation decision. It visualizes the efficient frontier, optimized ETF weights, and resulting portfolio exposures under explicit constraints on long-only weights, maximum ETF concentration, and the risk-free rate. In the reported solution, the maximum-Sharpe portfolio concentrates most heavily in `IVVW`, `IDV`, and `DVYA`.

<img src="docs/figures/Dashboard 5 Portfolio.png" alt="Portfolio Recommendation" width="900">

## Repository Structure

```text
.
|-- data/
|   |-- raw/                  # Source datasets gathered from external providers
|   `-- processed/            # Cleaned monthly tables used for analytics
|-- docs/
|   |-- ETF_allocation_support_report.pdf
|   `-- figures/              # Dashboard screenshots used in the report and README
|-- notebooks/                # Validation and exploratory notebooks
|-- scripts/                  # Thin CLI wrappers for data and model workflows
|-- src/
|   |-- downloaders/          # Provider-specific download logic
|   |-- gathering/            # CLI entry points for raw data collection
|   |-- llm/                  # Optional ETF classification workflow
|   |-- optimizer/            # Portfolio simulation and optimization logic
|   |-- preprocessing/        # Monthly transformations and feature engineering
|   |-- tableau_model/        # Tableau relational model builders
|   |-- config.py
|   |-- tableau_relational.py
|   `-- utils.py
|-- tableau_input/            # Tableau workbook and exported Excel inputs
|-- tests/                    # Pipeline and integration tests
|-- pyproject.toml
`-- README.md
```

## Project Commands

Install dependencies:

```powershell
poetry install
Copy-Item .env.example .env
```

Optional OpenAI configuration:

```text
OPENAI_API_KEY="your_openai_api_key"
OPENAI_MODEL="gpt-5.4-mini"
```

Run the full data pipeline:

```powershell
poetry run bi-fetch-etfs
poetry run bi-fetch-prices --start-date 2016-01-01
poetry run bi-fetch-macro --max-workers 4
poetry run bi-classify-etfs --mode auto --batch-threshold 50
poetry run bi-preprocess
poetry run bi-build-relational-tableau
poetry run bi-build-portfolio-optimizer --max-weight 0.3 --risk-free-rate 0.03 --etfs-list "DIVB,DVY,DVYA,HDV,IDV,IVVW,PFF"
```

Run tests:

```powershell
poetry run pytest tests
```
