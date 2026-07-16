# Hyperliquid Trader Analytics: Market Sentiment & Behavioral Profiling

A robust data pipeline and interactive Streamlit dashboard that uncovers how macroeconomic sentiment (Bitcoin Fear & Greed Index) drives algorithmic trader behavior, profitability, and risk management on the Hyperliquid DEX.

---

## 📂 Repository Structure

- `data/` : Directory for the raw CSV datasets (ignored by git, download details below).
  - `bitcoin_sentiment.csv` : Bitcoin Fear & Greed Index history.
  - `historical_trader_data.csv` : Detailed trade execution history.
- `processed/` : Contains generated outputs and metrics.
  - `daily_trader_metrics.csv` : Daily aggregated trader metrics.
  - `account_profiles.csv` : Aggregated trader profiles and clustering assignments.
  - `data_quality_anomalies.log` : Data quality audit log.
  - `plots/` : High-resolution charts (violin plots, box plots, K-Means clustering, fee analysis).
- `download_data.py` : Programmatic Google Drive downloader for raw CSV files.
- `analysis.py` : Core data pipeline (data cleaning, validation, clustering, predictive model, plot generation).
- `analysis.ipynb` : Documented Jupyter Notebook showing step-by-step analysis and plots.
- `dashboard.py` : Interactive Streamlit dashboard with a What-If simulator and data quality log viewer.
- `requirements.txt` : Python dependencies for environment replication.
- `.gitignore` : Configured to exclude IDE files, temporary files, and the large raw dataset.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher.
- In your terminal, run the following command to install the required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Step 1: Download the Datasets
Run the automated downloader to fetch the raw datasets:
```bash
python download_data.py
```

### Step 2: Run the Analysis Pipeline
Execute the pipeline script to run data cleaning, validate dataset quality, calculate behavior metrics, cluster accounts, train the predictive model, and output the charts:
```bash
python analysis.py
```
This will populate the `processed/` directory with the audit logs, CSVs, and PNG charts.

### Step 3: Run the Streamlit Dashboard
Launch the interactive web application to explore the clusters, view fee/volume analytics, and run next-day profit simulations:
```bash
streamlit run dashboard.py
```
This will start a local server at `http://localhost:8501` and open the app in your default browser.

---

## 💡 Key Design Choices

1. **Refined Long/Short Metrics**: 
   Standard long/short ratios often double-count trade executions (counting a long-close order as a short trade). To resolve this, behavior is split into:
   - **Taker Buy Ratio**: Buy trades vs. total trades (execution pressure).
   - **Long Opening Ratio**: Open Long trades vs. total opens (positioning intent).
   This split provides a cleaner, more accurate representation of trader sentiment.
2. **Cost & Volume Analytics**:
   Includes detailed tracking of fees and trading volume to analyze how algorithmic high-frequency trading costs eat into PnL.
3. **Data Quality & Anomaly Logging**:
   A robust data validation framework screens datasets upon ingestion for missing values, duplicate rows, and statistical outliers. All warnings are written to `processed/data_quality_anomalies.log`.

---

## 📊 Executive Analysis Report: Trader Performance vs. Market Sentiment

### 📝 Executive Summary
This analysis explores how Bitcoin market sentiment (Fear & Greed Index) relates to trader behavior and profitability on Hyperliquid. Using transaction data from active accounts, we examined how sentiment swings drive shifts in position sizing, margin modes, and trading frequencies.

Key findings show that traders engage in distinct contrarian behavior when opening new positions: they build longs during market panic (Long Opening Ratio: 53.2% on Fear days) and short overextended rallies (Long Opening Ratio: 47.9% on Greed days). However, this contrarian bias during Fear days introduces severe tail risk, with the 5th percentile daily PnL (drawdown proxy) dropping significantly compared to Greed days.

---

### 1. Key Insights & Analysis

#### 📉 Insight 1: The Volatility Premium and Tail Risk of Fear Days
Traders realize **25% higher average returns** on Fear days compared to Greed days. However, this premium is accompanied by catastrophic downside risk. The **drawdown proxy (5th percentile daily PnL)** is drastically worse during Fear regimes. This demonstrates that while panicking markets offer lucrative mean-reversion entries, they expose traders to high liquidation risk.

#### 🔄 Insight 2: Position-Level Contrarian Biases
By decoupling long/short behavior into *Taker Buy Ratio* and *Long Opening Ratio*, the data reveals a strong contrarian positioning intent: during **Fear** regimes, the median Long Opening Ratio is **53.2%** (dip-buying), whereas during **Greed** regimes, it drops to **47.9%** (anticipatory shorting). Traders aggressively fade macro extremes when establishing new exposure.

#### 👥 Insight 3: Behavioral Archetypes (K-Means Clustering)
Using K-Means clustering on trades count, average size, cross margin ratio, win rate, and long opening ratio, we identified 3 distinct trader archetypes:
* **Cluster 0 (High Long Openers)**: Cross margin heavy users who focus heavily on buying and opening new longs, trading with moderate sizes and low frequencies.
* **Cluster 1 (High-Volume Swing Traders)**: Highly active accounts using large trade sizes and balanced long opening ratios.
* **Cluster 2 (Retail Scalpers)**: Ultra-high frequency traders using mixed margin profiles.

#### 💰 Insight 4: Cost Analytics and Volume
Fee expenditure scales exponentially with trade frequency, severely impacting the net PnL of retail scalping clusters. Volume also surges during Fear days, indicating panic-induced liquidity.

---

### 2. Predictive Modeling Results
A **Random Forest Classifier** was trained to predict next-day profitability (Profit vs. Loss) using daily behavioral statistics, today's PnL, fees, volume, and the Bitcoin Fear & Greed Index value.
* **Feature Importances**:
  1. *Average Trade Size* & *Trades Count* remain the dominant features.
  2. *Total Volume* & *Total Fees* contribute heavily, indicating risk exposure and cost burden.
  3. *Bitcoin FGI Sentiment* confirms the macro environment's direct effect on algorithmic success.


