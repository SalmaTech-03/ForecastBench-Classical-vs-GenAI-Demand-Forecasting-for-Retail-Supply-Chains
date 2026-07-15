

<h1> ForecastBench</h1>
<h3>Classical vs GenAI Demand Forecasting for Retail Supply Chains</h3>

<p>
An end-to-end supply chain analytics pipeline that rigorously benchmarks a classical
time-series model against a generative pre-trained forecasting model — with an honest,
data-backed verdict, not an assumption.
</p>

<!-- BADGES: these render live from shields.io, no image files needed -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/geopandas-geospatial-brightgreen?logo=googlemaps&logoColor=white" alt="geopandas">
  <img src="https://img.shields.io/badge/StatsForecast-MSTL-orange" alt="StatsForecast">
  <img src="https://img.shields.io/badge/Nixtla-TimeGPT-purple" alt="TimeGPT">
  <img src="https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Why This Project](#-why-this-project)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Takeaway](#-key-takeaway)
- [Visualizations](#-visualizations)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## Overview

Most forecasting portfolios pick a model, run it once, and report an accuracy number. This
project does the opposite: it treats model selection as a question to be answered with
evidence, not assumed in advance.

**ForecastBench** segments retail SKUs by business value and demand variability, visualizes
sales geographically, and then puts a classical statistical model (**MSTL**) head-to-head
against a generative pre-trained time-series model (**TimeGPT**) — using rolling
cross-validation and multiple error metrics, including ones built specifically for
intermittent demand (stockout risk), not just MAE.

## Why This Project

| Business Question | What This Project Answers |
|---|---|
| Which products deserve the tightest inventory control? | ABC-XYZ classification matrix |
| Where is demand concentrated geographically? | Country and state-level choropleth maps |
| Can a GenAI forecasting model be trusted over a classical one? | Rolling-CV benchmark, both models, same data |
| How do we know a forecast is *reliable*, not just accurate on average? | CFE, PIS, and MAR — intermittent-demand metrics |

##  Dataset

> _Fill in once finalized:_
- **Source:** [dataset name / Kaggle link]
- **Size:** ~X rows, X columns
- **Granularity:** weekly demand per store-SKU combination
- **Time range:** [start] – [end]

## Methodology

```
Raw Data
   │
   ▼
① Data Cleaning & Consistency Checks  ──► resolved market/year coverage gaps
   │
   ▼
② Geospatial Analysis  ──► country & state-level sales distribution
   │
   ▼
③ ABC-XYZ Inventory Segmentation  ──► 9-class value × variability matrix
   │
   ▼
④ Demand Forecasting
   ├── Stationarity testing (ADF test)
   ├── Model A: MSTL (statistical decomposition + AutoARIMA trend)
   └── Model B: TimeGPT (generative pretrained, zero-shot + exogenous vars)
   │
   ▼
⑤ Evaluation  ──► MAE · SMAPE · CFE · PIS · MAR · rolling-window CV
   │
   ▼
⑥ Experiment Tracking  ──► MLflow (metrics, params, model artifacts)
```

## 📊 Results

<!-- Replace with your real numbers + real screenshots of your plots -->

| Model       | MAE ↓ | SMAPE ↓ | Cross-Val MAE ↓ |
|-------------|:-----:|:-------:|:----------------:|
| MSTL        | X.XX  | X.X%    | X.XX             |
| TimeGPT     | X.XX  | X.X%    | X.XX             |

<p align="center">
  <img src="assets/mae_comparison.png" alt="MAE comparison chart" width="600">
  <br><em>Fig 1. MAE comparison across models per store-SKU series</em>
</p>

<p align="center">
  <img src="assets/error_distribution.png" alt="Residual error distribution" width="600">
  <br><em>Fig 2. Residual distribution — tighter spread around zero indicates more reliable forecasts</em>
</p>

##  Key Takeaway

> _Write this once you have your real result — be specific and honest, e.g.:_
> "The classical MSTL model outperformed TimeGPT on this dataset, achieving X% lower MAE.
> This suggests that for well-structured, seasonal retail demand with enough historical
> depth, a properly tuned classical decomposition model can still beat a zero-shot
> foundation model — foundation models show their advantage more clearly on sparse or
> irregular series with little history."

## 🗺 Visualizations

<p align="center">
  <img src="assets/world_sales_map.png" alt="World sales choropleth" width="480">
  <img src="assets/us_store_map.png" alt="US store distribution map" width="480">
</p>
<p align="center"><em>Fig 3–4. Geographic sales concentration and store density</em></p>

<p align="center">
  <img src="assets/abc_xyz_matrix.png" alt="ABC-XYZ classification chart" width="600">
  <br><em>Fig 5. Revenue contribution by ABC-XYZ inventory class</em>
</p>

## 🛠 Tech Stack

`Python` · `pandas` · `geopandas` · `matplotlib` / `plotly` · `statsforecast` (MSTL, AutoARIMA)
· `Nixtla TimeGPT` · `scikit-learn` · `MLflow` · `Jupyter Notebook`

##  How to Run

```bash
git clone https://github.com/<your-username>/forecastbench.git
cd forecastbench
pip install -r requirements.txt

# Add your Nixtla API key to a .env file:
echo "NIXTLA_API_KEY=your_key_here" > .env

jupyter notebook ForecastBench.ipynb
```

##  Project Structure

```
forecastbench/
├── assets/                  # banner, chart exports for README
├── data/                    # raw + processed datasets
├── notebooks/
│   └── ForecastBench.ipynb  # main analysis notebook
├── src/                     # (optional) reusable functions if you modularize
├── requirements.txt
├── .env.example
└── README.md
```

---

<div align="center">
<sub>Built by <a href="#">Your Name</a> · connect on <a href="#">LinkedIn</a></sub>
</div>
