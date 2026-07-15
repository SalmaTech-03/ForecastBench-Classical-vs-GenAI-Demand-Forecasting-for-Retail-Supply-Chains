<div align="center">

<h1> ForecastBench</h1>
<h3>Classical vs GenAI Demand Forecasting for Retail Supply Chains</h3>
<p><em>A rigorous, evidence-first benchmark — not another notebook that picks a model and calls it done.</em></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/geopandas-geospatial-brightgreen?logo=googlemaps&logoColor=white" alt="geopandas">
  <img src="https://img.shields.io/badge/StatsForecast-MSTL-orange" alt="StatsForecast">
  <img src="https://img.shields.io/badge/Nixtla-TimeGPT-purple" alt="TimeGPT">
  <img src="https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="build">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

</div>

---

### TL;DR

> Forecasted weekly demand for hundreds of store-SKU combinations, benchmarked a classical
> decomposition model against a generative pretrained time-series model under identical
> rolling cross-validation, and let the data — not intuition — decide the winner. Also
> segmented inventory by value and demand volatility, mapped sales geographically, and
> tracked every experiment in MLflow.

---

##  Table of Contents

- [Problem Statement](#-problem-statement)
- [Why This Project Is Different](#-why-this-project-is-different)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Takeaway](#-key-takeaway)
- [Skills Demonstrated](#-skills-demonstrated)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [What I'd Do With More Time](#-what-id-do-with-more-time)

---

##  Problem Statement

Retailers lose money two ways: **stockouts** (lost sales, unhappy customers) and
**overstocking** (tied-up capital, waste). Both come down to the same root cause —
demand forecasts that are either too coarse to act on, or too fragile to trust in
production. This project asks a specific, answerable question:

> **For real-world, seasonal retail demand data, does a modern generative time-series
> model actually outperform a well-tuned classical statistical model — or is that
> assumption untested hype?**

##  Why This Project Is Different

Most forecasting portfolios do one of two things: pick a model and report an accuracy
number, or pick the newest model because it's newest. This project does neither.

- **No single train/test split.** Every model is evaluated with rolling-window
  cross-validation across 8 windows, because a single split can flatter or sabotage
  a model by luck.
- **No cherry-picked metric.** MAE and SMAPE are reported alongside intermittent-demand
  metrics — Cumulative Forecast Error (CFE), Periods-In-Stock (PIS), and Mean Absolute
  Rate (MAR) — because a low average error can still hide frequent stockouts.
- **No assumed winner.** The generative model (TimeGPT) was given exogenous variables
  and fine-tuning steps to give it a fair shot against the classical model (MSTL). The
  result is reported honestly either way.
- **Every experiment is logged**, not just the final one — via MLflow — so the
  comparison is reproducible, not anecdotal.

##  Dataset

> _Fill in once finalized:_
- **Source:** [dataset name / link]
- **Size:** ~X rows · X store-SKU series
- **Granularity:** weekly units sold per store-SKU combination
- **Time range:** [start] – [end]

##  Pipeline Architecture

```
                         ┌─────────────────────────┐
                         │      Raw Retail Data     │
                         └────────────┬────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
      ┌─────────────────┐  ┌───────────────────┐  ┌──────────────────┐
      │  Data Cleaning   │  │  Geospatial Layer  │  │  ABC-XYZ Segment  │
      │  & Consistency   │  │  (country / state   │  │  (value × demand  │
      │      Checks      │  │   choropleths)      │  │   variability)    │
      └────────┬─────────┘  └─────────┬──────────┘  └─────────┬─────────┘
               │                       │                       │
               └───────────────────────┼───────────────────────┘
                                       ▼
                         ┌─────────────────────────┐
                         │   Stationarity Testing    │
                         │      (ADF per series)      │
                         └────────────┬────────────┘
                                      ▼
                  ┌───────────────────┴───────────────────┐
                  ▼                                        ▼
        ┌───────────────────┐                  ┌────────────────────┐
        │   Model A: MSTL    │                  │  Model B: TimeGPT   │
        │  (LOESS decomp +   │                  │  (generative pre-   │
        │    AutoARIMA)      │                  │   trained, 0-shot)  │
        └─────────┬──────────┘                  └──────────┬─────────┘
                  └───────────────────┬──────────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │   Rolling-CV Evaluation (8 windows) │
                    │  MAE · SMAPE · CFE · PIS · MAR       │
                    └────────────────┬──────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │   MLflow Tracking     │
                          └─────────────────────┘
```

##  Methodology

| Stage | What Happens | Why It Matters |
|---|---|---|
| **1. Cleaning** | Drop PII/redundant fields, resolve year-over-year market coverage gaps | Garbage in, garbage out — most forecasting errors trace back here |
| **2. Geospatial** | Country + US-state sales choropleths via `geopandas` spatial joins | Surfaces regional demand concentration invisible in tabular summaries |
| **3. ABC-XYZ** | Classify SKUs by cumulative revenue (A/B/C) × demand variability (X/Y/Z) | Tells you *where* forecasting effort is worth spending |
| **4. Stationarity** | Augmented Dickey-Fuller test per series, differencing where needed | Prevents silently invalid statistical assumptions |
| **5. Forecasting** | MSTL vs TimeGPT, both with identical horizon and exogenous inputs | Fair, apples-to-apples benchmark |
| **6. Evaluation** | MAE, SMAPE, CFE, PIS, MAR, 8-window rolling CV | Measures both accuracy *and* reliability under stockout risk |
| **7. Tracking** | MLflow logs params, metrics, model artifacts per run | Reproducibility — a forecast without a logged run is a guess |

##  Results

| Model       | MAE ↓ | SMAPE ↓ | Cross-Val MAE ↓ | Cross-Val Std ↓ |
|-------------|:-----:|:-------:|:----------------:|:----------------:|
| MSTL        | X.XX  | X.X%    | X.XX             | X.XX             |
| TimeGPT     | X.XX  | X.X%    | X.XX             | X.XX             |

##  Key Takeaway

> _Fill in with your real, specific finding, e.g.:_
> "MSTL outperformed TimeGPT by X% on cross-validated MAE, with a tighter and more
> consistent error distribution. This suggests that for well-structured, moderately
> seasonal retail series with 2+ years of history, a properly tuned classical model
> remains more reliable than a zero-shot foundation model — the generative model's
> advantage likely emerges more clearly on sparse or irregular series with little
> historical depth, which is a distinction worth testing in future iterations."

##  Skills Demonstrated

`Time-series forecasting` · `Statistical modeling (MSTL, ARIMA)` · `Generative AI / LLM-based forecasting (TimeGPT)`
`Geospatial analysis` · `Inventory optimization` · `Rolling cross-validation design`
`Production ML practices (MLflow)` · `Honest model evaluation under multiple metrics`

##  Tech Stack

`Python` · `pandas` · `geopandas` · `statsforecast` (MSTL, AutoARIMA) · `Nixtla TimeGPT`
`scikit-learn` · `MLflow` · `Jupyter Notebook`

##  How to Run

```bash
git clone https://github.com/<your-username>/forecastbench.git
cd forecastbench
pip install -r requirements.txt

echo "NIXTLA_API_KEY=your_key_here" > .env

jupyter notebook ForecastBench.ipynb
```

##  Project Structure

```
forecastbench/
├── data/                    # raw + processed datasets
├── notebooks/
│   └── ForecastBench.ipynb  # main analysis notebook
├── src/                     # reusable functions (optional modularization)
├── requirements.txt
├── .env.example
└── README.md
```

##  What I'd Do With More Time

- Test both models on a sparser, lower-history dataset to see if TimeGPT's advantage
  emerges under different data conditions
- Add promotional and pricing events as additional exogenous variables
- Wrap the winning model behind a lightweight FastAPI endpoint for real-time inference
- Extend the geospatial layer into a routing/logistics-cost overlay

---

