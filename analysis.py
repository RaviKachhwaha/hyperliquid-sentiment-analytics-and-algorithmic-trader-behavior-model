import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def validate_and_log_anomalies(sentiment, trader):
    """Performs rigorous data quality checks and logs findings to processed/data_quality_anomalies.log."""
    os.makedirs('processed', exist_ok=True)
    log_path = 'processed/data_quality_anomalies.log'
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write("             DATA QUALITY & ANOMALY VALIDATION LOG          \n")
        f.write("============================================================\n\n")
        
        # 1. Dataset Dimensions
        f.write("### 1. DATASET DIMENSIONS\n")
        f.write(f"- Bitcoin Sentiment Dataset: {sentiment.shape[0]} rows, {sentiment.shape[1]} columns\n")
        f.write(f"- Historical Trader Dataset (Hyperliquid): {trader.shape[0]} rows, {trader.shape[1]} columns\n\n")
        
        # 2. Missing Values Check
        f.write("### 2. MISSING VALUES ANALYSIS\n")
        s_missing = sentiment.isnull().sum()
        t_missing = trader.isnull().sum()
        
        f.write("  Sentiment Dataset Missing Values:\n")
        for col, count in s_missing.items():
            f.write(f"    - {col}: {count}\n")
            
        f.write("  Trader Dataset Missing Values:\n")
        for col, count in t_missing.items():
            f.write(f"    - {col}: {count}\n")
        f.write("\n")
        
        # 3. Duplicate Rows Check
        f.write("### 3. DUPLICATE ROWS ANALYSIS\n")
        s_dups = sentiment.duplicated().sum()
        t_dups = trader.duplicated().sum()
        f.write(f"  - Duplicate rows in Sentiment dataset: {s_dups}\n")
        f.write(f"  - Duplicate rows in Trader dataset: {t_dups}\n\n")
        
        # 4. Date Coverage & Alignment
        f.write("### 4. DATE ALIGNMENT & COVERAGE\n")
        sentiment_dates = pd.to_datetime(sentiment['date'])
        trader_dates = pd.to_datetime(trader['Timestamp IST'], dayfirst=True).dt.normalize()
        
        f.write(f"  - Sentiment date range: {sentiment_dates.min().strftime('%Y-%m-%d')} to {sentiment_dates.max().strftime('%Y-%m-%d')}\n")
        f.write(f"  - Trader data date range: {trader_dates.min().strftime('%Y-%m-%d')} to {trader_dates.max().strftime('%Y-%m-%d')}\n")
        
        # Check alignment gap
        sentiment_set = set(sentiment_dates.dt.date)
        trader_set = set(trader_dates.dt.date)
        missing_sentiment_dates = trader_set - sentiment_set
        f.write(f"  - Trader dates missing from Sentiment Index: {len(missing_sentiment_dates)} days\n")
        if len(missing_sentiment_dates) > 0:
            f.write(f"    Missing dates: {sorted(list(missing_sentiment_dates))[:5]}...\n")
        f.write("\n")
        
        # 5. Outlier Detection (IQR Method)
        f.write("### 5. STATISTICAL OUTLIER DETECTION (IQR Method)\n")
        # Outliers in PnL
        q1_pnl = trader['Closed PnL'].quantile(0.25)
        q3_pnl = trader['Closed PnL'].quantile(0.75)
        iqr_pnl = q3_pnl - q1_pnl
        lower_pnl = q1_pnl - 3 * iqr_pnl
        upper_pnl = q3_pnl + 3 * iqr_pnl
        
        pnl_outliers = trader[(trader['Closed PnL'] < lower_pnl) | (trader['Closed PnL'] > upper_pnl)]
        pnl_pct = (len(pnl_outliers) / len(trader)) * 100
        f.write(f"  Closed PnL Outliers (Bounds: ${lower_pnl:.2f} to ${upper_pnl:.2f}):\n")
        f.write(f"    - Number of extreme outlier trades: {len(pnl_outliers)} ({pnl_pct:.2f}% of total)\n")
        
        # Outliers in Trade Size
        q1_size = trader['Size USD'].quantile(0.25)
        q3_size = trader['Size USD'].quantile(0.75)
        iqr_size = q3_size - q1_size
        lower_size = q1_size - 3 * iqr_size
        upper_size = q3_size + 3 * iqr_size
        
        size_outliers = trader[(trader['Size USD'] < lower_size) | (trader['Size USD'] > upper_size)]
        size_pct = (len(size_outliers) / len(trader)) * 100
        f.write(f"  Trade Size USD Outliers (Bounds: ${lower_size:.2f} to ${upper_size:.2f}):\n")
        f.write(f"    - Number of extreme outlier trades: {len(size_outliers)} ({size_pct:.2f}% of total)\n")
        
        # Top outlier accounts
        top_pnl_outliers = pnl_outliers['Account'].value_counts().head(3)
        f.write("  Top 3 Accounts generating extreme PnL outliers:\n")
        for acc, count in top_pnl_outliers.items():
            f.write(f"    - Account {acc[:10]}...: {count} outlier trades\n")
            
        f.write("\n============================================================\n")
        
    print(f"  Data validation complete. Quality & anomaly logs written to {log_path}")

def load_and_preprocess_data():
    """Loads datasets and performs cleaning and date alignment."""
    print("Step 1: Ingesting datasets...")
    sentiment_path = 'data/bitcoin_sentiment.csv'
    trader_path = 'data/historical_trader_data.csv'
    
    if not os.path.exists(sentiment_path) or not os.path.exists(trader_path):
        raise FileNotFoundError("Missing raw data files in data/ directory. Please run the download script first.")
        
    sentiment = pd.read_csv(sentiment_path)
    trader = pd.read_csv(trader_path)
    
    # Run data quality validation & logging
    validate_and_log_anomalies(sentiment, trader)
    
    # 2. Convert timestamps and extract date (YYYY-MM-DD)
    sentiment['date'] = pd.to_datetime(sentiment['date'])
    
    # Parse precise datetimes from Timestamp IST string field
    trader['datetime'] = pd.to_datetime(trader['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trader['date'] = trader['datetime'].dt.normalize()
    
    # 3. Categorize sentiment classification into Fear vs Greed
    def categorize(val):
        if 'Fear' in val:
            return 'Fear'
        elif 'Greed' in val:
            return 'Greed'
        else:
            return 'Neutral'
            
    sentiment['sentiment_category'] = sentiment['classification'].apply(categorize)
    
    # Merge datasets
    merged = pd.merge(
        trader, 
        sentiment[['date', 'value', 'classification', 'sentiment_category']], 
        on='date', 
        how='inner'
    )
    print(f"  Aligned dataset shape: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return sentiment, trader, merged

def compute_key_metrics(merged):
    """Computes key trader metrics aggregated daily."""
    print("\nStep 2: Computing daily trader metrics...")
    
    # Helper indicators
    merged['is_win'] = merged['Closed PnL'] > 0
    merged['is_loss'] = merged['Closed PnL'] < 0
    merged['is_trade'] = merged['Closed PnL'] != 0
    
    # Refined metrics: Mutually exclusive buy/sell and open-long/open-short
    merged['is_buy'] = merged['Side'] == 'BUY'
    merged['is_sell'] = merged['Side'] == 'SELL'
    merged['is_open_long'] = merged['Direction'] == 'Open Long'
    merged['is_open_short'] = merged['Direction'] == 'Open Short'
    
    # Aggregate daily per trader (account)
    daily = merged.groupby(['Account', 'date']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        total_trades=('Trade ID', 'count'),
        closed_trades=('is_trade', 'sum'),
        winning_trades=('is_win', 'sum'),
        total_size_usd=('Size USD', 'sum'),
        avg_size_usd=('Size USD', 'mean'),
        cross_margin_trades=('Crossed', 'sum'),
        buy_trades=('is_buy', 'sum'),
        sell_trades=('is_sell', 'sum'),
        open_long_trades=('is_open_long', 'sum'),
        open_short_trades=('is_open_short', 'sum'),
        sentiment_val=('value', 'first'),
        sentiment_class=('classification', 'first'),
        sentiment_cat=('sentiment_category', 'first'),
        total_fees=('Fee', 'sum'),
        top_coin=('Coin', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
    ).reset_index()
    
    # Calculate ratios
    daily['win_rate'] = np.where(daily['closed_trades'] > 0, daily['winning_trades'] / daily['closed_trades'], 0.0)
    daily['cross_margin_ratio'] = daily['cross_margin_trades'] / daily['total_trades']
    daily['buy_ratio'] = daily['buy_trades'] / daily['total_trades']
    daily['fee_to_pnl_ratio'] = np.where(daily['daily_pnl'] > 0, daily['total_fees'] / daily['daily_pnl'], 0.0)
    
    # Long open ratio represents the share of new positions that are long
    daily['long_open_ratio'] = np.where(
        (daily['open_long_trades'] + daily['open_short_trades']) > 0,
        daily['open_long_trades'] / (daily['open_long_trades'] + daily['open_short_trades']),
        0.5
    )
    
    os.makedirs('processed', exist_ok=True)
    daily.to_csv('processed/daily_trader_metrics.csv', index=False)
    print("  Daily metrics computed and saved to processed/daily_trader_metrics.csv")
    return daily

def run_segmentation(merged):
    """Segments traders using behavioral rules and K-Means clustering."""
    print("\nStep 3: Running trader segmentation & clustering...")
    
    merged['is_win'] = merged['Closed PnL'] > 0
    merged['is_trade'] = merged['Closed PnL'] != 0
    merged['is_buy'] = merged['Side'] == 'BUY'
    merged['is_open_long'] = merged['Direction'] == 'Open Long'
    merged['is_open_short'] = merged['Direction'] == 'Open Short'
    
    # Group at the Account (trader) level
    account_df = merged.groupby('Account').agg(
        total_pnl=('Closed PnL', 'sum'),
        total_trades=('Trade ID', 'count'),
        closed_trades=('is_trade', 'sum'),
        winning_trades=('is_win', 'sum'),
        avg_trade_size=('Size USD', 'mean'),
        total_volume=('Size USD', 'sum'),
        cross_margin_ratio=('Crossed', 'mean'),
        buy_ratio=('is_buy', 'mean'),
        open_longs=('is_open_long', 'sum'),
        open_shorts=('is_open_short', 'sum'),
        total_fees=('Fee', 'sum')
    ).reset_index()
    
    account_df['win_rate'] = np.where(account_df['closed_trades'] > 0, account_df['winning_trades'] / account_df['closed_trades'], 0.0)
    
    # Calculate long open ratio for overall account profile
    account_df['long_open_ratio'] = np.where(
        (account_df['open_longs'] + account_df['open_shorts']) > 0,
        account_df['open_longs'] / (account_df['open_longs'] + account_df['open_shorts']),
        0.5
    )
    
    # Rule-based segments
    account_df['freq_segment'] = pd.cut(
        account_df['total_trades'],
        bins=[0, 100, 1000, np.inf],
        labels=['Infrequent (<100)', 'Moderate (100-1000)', 'Frequent (>1000)']
    )
    
    account_df['margin_segment'] = pd.cut(
        account_df['cross_margin_ratio'],
        bins=[-0.01, 0.3, 0.7, 1.01],
        labels=['Isolated Margin Heavy (<0.3)', 'Mixed Margin (0.3-0.7)', 'Cross Margin Heavy (>0.7)']
    )
    
    account_df['profit_segment'] = np.where(
        (account_df['total_pnl'] > 0) & (account_df['win_rate'] > 0.5), 
        'Consistent Winner',
        np.where(account_df['total_pnl'] < 0, 'Losing/Inconsistent', 'Neutral/Slight Winner')
    )
    
    # KMeans Clustering on behavior features
    features = ['total_trades', 'avg_trade_size', 'cross_margin_ratio', 'win_rate', 'long_open_ratio']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(account_df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    account_df['cluster'] = kmeans.fit_predict(scaled)
    
    account_df.to_csv('processed/account_profiles.csv', index=False)
    print("  Account profiles and behavioral clustering saved to processed/account_profiles.csv")
    return account_df

def run_predictive_model(daily):
    """Trains a Random Forest classifier to predict next-day profitability using lagged behaviors."""
    print("\nStep 4: Training predictive model for next-day profitability...")
    
    daily_sorted = daily.sort_values(by=['Account', 'date'])
    daily_sorted['next_day_pnl'] = daily_sorted.groupby('Account')['daily_pnl'].shift(-1)
    daily_sorted['next_day_profit'] = (daily_sorted['next_day_pnl'] > 0).astype(int)
    
    model_data = daily_sorted.dropna(subset=['next_day_pnl']).copy()
    
    features = ['daily_pnl', 'total_trades', 'win_rate', 'avg_size_usd', 'cross_margin_ratio', 'buy_ratio', 'long_open_ratio', 'sentiment_val', 'total_fees', 'total_size_usd']
    X = model_data[features]
    y = model_data['next_day_profit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  Model Accuracy: {accuracy:.4f}")
    
    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv('processed/model_feature_importances.csv')
    print("  Feature importances saved to processed/model_feature_importances.csv")
    return rf, features

def generate_visualizations(daily, account_df):
    """Generates clean charts for reporting and dashboard use."""
    print("\nStep 5: Generating analysis plots...")
    sns.set_theme(style="whitegrid")
    os.makedirs('processed/plots', exist_ok=True)
    
    # Filter out neutral days for distinct Fear vs Greed comparison
    fg_daily = daily[daily['sentiment_cat'].isin(['Fear', 'Greed'])].copy()
    
    # Plot 1: Daily PnL Distribution (Clipped for visual quality)
    plt.figure(figsize=(10, 6))
    lower = fg_daily['daily_pnl'].quantile(0.02)
    upper = fg_daily['daily_pnl'].quantile(0.98)
    clipped = fg_daily[(fg_daily['daily_pnl'] >= lower) & (fg_daily['daily_pnl'] <= upper)]
    sns.violinplot(x='sentiment_cat', y='daily_pnl', data=clipped, palette='coolwarm', hue='sentiment_cat', legend=False)
    plt.title('Daily Trader PnL Distribution by Sentiment (96% Core Values)')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Daily PnL (USD)')
    plt.tight_layout()
    plt.savefig('processed/plots/pnl_distribution.png', dpi=300)
    plt.close()
    
    # Plot 2: Behavioral Shifts (Taker Buy Ratio, Long Opening Ratio & Cross Margin Ratio)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    sns.boxplot(x='sentiment_cat', y='buy_ratio', data=fg_daily, ax=axes[0], palette='coolwarm', hue='sentiment_cat', legend=False)
    axes[0].set_title('Taker Buy Ratio Shifts')
    axes[0].set_xlabel('Market Sentiment')
    axes[0].set_ylabel('Buy Trades / Total Trades')
    
    sns.boxplot(x='sentiment_cat', y='long_open_ratio', data=fg_daily, ax=axes[1], palette='coolwarm', hue='sentiment_cat', legend=False)
    axes[1].set_title('Long Opening Ratio Shifts (New Positions)')
    axes[1].set_xlabel('Market Sentiment')
    axes[1].set_ylabel('Open Longs / Total Opens')
    
    sns.boxplot(x='sentiment_cat', y='cross_margin_ratio', data=fg_daily, ax=axes[2], palette='coolwarm', hue='sentiment_cat', legend=False)
    axes[2].set_title('Cross Margin Exposure Shifts')
    axes[2].set_xlabel('Market Sentiment')
    axes[2].set_ylabel('Cross Margin Trades / Total Trades')
    
    plt.suptitle('Trader Behavioral Modifications Under Sentiment Shift')
    plt.tight_layout()
    plt.savefig('processed/plots/behavioral_shifts.png', dpi=300)
    plt.close()
    
    # Plot 3: Drawdown vs Average Returns (Risk/Return Tradeoff)
    avg_pnl = fg_daily.groupby('sentiment_cat')['daily_pnl'].mean()
    tail_risk = fg_daily.groupby('sentiment_cat')['daily_pnl'].quantile(0.05)
    
    tradeoff = pd.DataFrame({
        'Average Return': avg_pnl,
        'Tail Drawdown (5th Pct)': tail_risk
    }).reset_index()
    
    melted = pd.melt(tradeoff, id_vars='sentiment_cat', value_vars=['Average Return', 'Tail Drawdown (5th Pct)'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sentiment_cat', y='value', hue='variable', data=melted, palette='muted')
    plt.title('Risk-Return Tradeoff: Fear vs Greed Days')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Value (USD)')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('processed/plots/risk_return_tradeoff.png', dpi=300)
    plt.close()
    
    # Plot 4: K-Means Trader Archetypes
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='avg_trade_size', 
        y='total_trades', 
        hue='cluster', 
        style='profit_segment',
        size='win_rate',
        sizes=(30, 250),
        data=account_df, 
        palette='Set1',
        alpha=0.85
    )
    plt.title('Trader Behavioral Archetypes (K-Means Clustering)')
    plt.xlabel('Average Trade Size (USD)')
    plt.ylabel('Total Trades executed')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('processed/plots/trader_clusters.png', dpi=300)
    plt.close()
    
    # Plot 5: Fee Impact vs PnL
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='total_fees',
        y='daily_pnl',
        hue='sentiment_cat',
        size='total_size_usd',
        data=fg_daily,
        palette='coolwarm',
        alpha=0.7
    )
    plt.title('Fee Expenditure vs Daily PnL')
    plt.xlabel('Total Daily Fees (USD)')
    plt.ylabel('Daily Net PnL (USD)')
    plt.tight_layout()
    plt.savefig('processed/plots/fee_vs_pnl.png', dpi=300)
    plt.close()
    
    # Plot 6: Volume by Sentiment
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sentiment_cat', y='total_size_usd', data=fg_daily, estimator=sum, errorbar=None, palette='muted')
    plt.title('Total Traded Volume by Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Total Volume (USD)')
    plt.tight_layout()
    plt.savefig('processed/plots/volume_by_sentiment.png', dpi=300)
    plt.close()
    
    print("  Visualizations generated and saved to processed/plots/")

def main():
    print("=" * 60)
    print("Starting Trader Performance vs Sentiment Pipeline")
    print("=" * 60)
    
    # Run pipeline steps
    sentiment, trader, merged = load_and_preprocess_data()
    daily = compute_key_metrics(merged)
    account_df = run_segmentation(merged)
    rf, features = run_predictive_model(daily)
    generate_visualizations(daily, account_df)
    
    print("\nPipeline executed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
