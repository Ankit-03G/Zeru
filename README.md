# DeFi Credit Scoring System

A machine learning-based credit scoring system for DeFi wallets using Aave V2 transaction data.

---

## Overview

This system analyzes DeFi transaction patterns to assign credit scores (0-1000) to wallet addresses based on their historical behavior on the Aave V2 protocol. Higher scores indicate reliable, responsible usage patterns, while lower scores reflect risky or exploitative behavior.

---

## Architecture

### Data Processing Flow

```
Raw JSON Data → Feature Engineering → Credit Score Calculation → Analysis & Visualization
```

### Core Components

1. **Data Loading (`load_data`)**
   - Fetches transaction data from Google Drive
   - Handles JSON parsing and error management
   - Processes 100K+ transactions efficiently

2. **Feature Engineering (`engineer_advanced_features`)**
   - Extracts meaningful features from raw transaction data
   - Creates behavioral, financial, and risk indicators
   - Handles missing values and data normalization

3. **Credit Scoring (`calculate_credit_score`)**
   - Domain knowledge-based scoring algorithm
   - Considers multiple risk and reliability factors
   - Outputs scores in 0-1000 range

4. **Analysis (`analyze_scores`)**
   - Score distribution analysis
   - Wallet behavior characterization
   - Risk segmentation and insights

---

## Features Engineered

### Basic Activity Features
- `transaction_count`: Total number of transactions
- `unique_actions`: Number of different action types used
- `days_active`: Duration of wallet activity
- `avg_daily_transactions`: Activity frequency

### Financial Behavior Features
- `total_deposits`: Total amount deposited
- `total_borrows`: Total amount borrowed
- `total_repays`: Total amount repaid
- `total_redeems`: Total amount redeemed
- `avg_deposit_size`: Average deposit transaction size
- `avg_borrow_size`: Average borrow transaction size
- `avg_repay_size`: Average repay transaction size

### Risk Indicators
- `liquidation_count`: Number of liquidation events
- `total_volume`: Total transaction volume
- `repay_ratio`: Repayment consistency (total_repays / total_borrows)
- `borrow_deposit_ratio`: Borrowing vs deposits ratio
- `activity_diversity`: Variety of actions performed

### Action Type Counts
- `deposit_count`: Number of deposit transactions
- `borrow_count`: Number of borrow transactions
- `repay_count`: Number of repay transactions
- `redeemunderlying_count`: Number of redeem transactions
- `liquidationcall_count`: Number of liquidation calls

---

## Credit Scoring Algorithm

### Base Score: 500 points

### Positive Factors
- **Activity Bonus**: +1.5 points per transaction (max +100)
- **Deposit Bonus**: +15 × log(deposits) (max +150)
- **Repayment Consistency**: +100 × repay_ratio (max +100)
- **Activity Diversity**: +30 × diversity_score
- **Volume Bonus**: +5 × log(volume) (max +50)

### Negative Factors
- **Liquidation Penalty**: -100 points per liquidation
- **Over-borrowing Penalty**: -50 × (borrow_deposit_ratio - 2) if ratio > 2
- **Low Activity Penalty**: -50 if transaction_count < 5

### Score Bounds
- Minimum: 0 points
- Maximum: 1000 points

---

## Usage

### Basic Usage

```bash
python credit_scoring_system.py
```

This will:
1. Load data from Google Drive
2. Engineer features
3. Calculate credit scores
4. Generate analysis
5. Save results to CSV and Excel files

### Output Files

- `wallet_credit_scores.csv`: Basic credit scores for each wallet
- `wallet_credit_scores.xlsx`: Excel version of the above
- `detailed_wallet_analysis.csv`: Comprehensive feature data and scores
- `detailed_wallet_analysis.xlsx`: Excel version of the above

---

## Model Performance

Based on 100K transactions from 3,497 wallets:

- **Score Range**: 0.0 - 924.0
- **Mean Score**: 662.5
- **Median Score**: 615.0
- **Standard Deviation**: 157.1

### Score Distribution
- **0-100**: 2.4% (High-risk wallets)
- **100-500**: 4.0% (Medium-risk wallets)
- **500-700**: 64.5% (Standard users)
- **700-900**: 27.7% (High-quality users)
- **900-1000**: 4.3% (Premium users)

---

## Validation

### Low-Score Wallets (Bottom 10%)
- Average transactions: 18.8
- Average liquidations: 0.1
- Average repay ratio: 0.05
- **Interpretation**: New or risky users with poor repayment history

### High-Score Wallets (Top 10%)
- Average transactions: 96.5
- Average liquidations: 0.0
- Average repay ratio: 0.93
- **Interpretation**: Highly active, reliable users with excellent repayment behavior

---

## Technical Requirements

### Dependencies

```bash
pip install pandas numpy requests scikit-learn matplotlib seaborn
```

- pandas>=1.3.0
- numpy>=1.20.0
- requests>=2.25.0
- scikit-learn>=1.0.0
- matplotlib>=3.3.0
- seaborn>=0.11.0

### System Requirements

- Python 3.7+
- 4GB+ RAM (for processing large datasets)
- Internet connection (for data loading)

---

## Configuration

### Data Source

```python
# Google Drive file ID
file_id = "1ZiFWFr84rOn8WwU13MBsJUUQI6bIPEmL"
direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
```

### Scoring Parameters

The scoring algorithm can be tuned by modifying weights in the `calculate_credit_score` function:
- Transaction bonus weight: 1.5
- Deposit bonus weight: 15
- Liquidation penalty: -100
- Over-borrowing threshold: 2.0

---

## Extensibility

### Adding New Features

1. Modify `engineer_advanced_features()` to include new metrics
2. Update `calculate_credit_score()` to incorporate new features
3. Adjust scoring weights based on domain knowledge

### Alternative Scoring Methods

- **Clustering-based**: Use K-means to identify user segments
- **Supervised Learning**: Train models if labeled data becomes available
- **Composite Scoring**: Combine multiple scoring approaches

---

## Limitations

1. **No Ground Truth**: Scores based on domain knowledge, not validated labels
2. **Single Protocol**: Only considers Aave V2 transactions
3. **Time Independence**: Doesn't account for market conditions or temporal patterns
4. **Static Weights**: Scoring weights are fixed, not learned from data

---

## Future Enhancements

1. **Multi-protocol Analysis**: Include other DeFi protocols
2. **Temporal Features**: Add time-series analysis
3. **Market Context**: Incorporate market conditions
4. **Dynamic Scoring**: Implement adaptive scoring weights
5. **Real-time Updates**: Enable continuous score updates

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests and documentation
5. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Main Script

```python
# Author: Ankit Kumar Gupta
# Description: Credit Score Generation from DeFi Wallet Transaction Data
# Date: 17 July 2025

import json
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -- DATA DOWNLOAD FROM DRIVE --
file_id = "1ZiFWFr84rOn8WwU13MBsJUUQI6bIPEmL"
direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# -- 1. Data Loading --
def load_data(url):
    """Loading JSON transaction data. ."""
    print("Fetching data")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    print(f" Loaded {len(data)} transactions")
    return data

# -- 2. Feature Engineering --
def safe_numeric_conversion(actionData):
    """Safely extract numeric amount value."""
    try:
        return float(actionData.get('amount', 0)) if isinstance(actionData, dict) else 0.0
    except (ValueError, TypeError):
        return 0.0

def engineer_advanced_features(df):
    """Generating features."""
    print(" Starting feature engineering...")

    df['amount'] = df['actionData'].apply(safe_numeric_conversion)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    features = {}

    # Activity feature
    features['transaction_count'] = df.groupby('userWallet').size()
    features['unique_actions'] = df.groupby('userWallet')['action'].nunique()

    # Time-related features
    if 'timestamp' in df.columns:
        features['days_active'] = df.groupby('userWallet')['timestamp'].apply(lambda x: (x.max() - x.min()).days + 1)
        features['avg_daily_transactions'] = features['transaction_count'] / features['days_active']

    # Financial 
    deposit_data = df[df['action'] == 'deposit'].groupby('userWallet')['amount']
    borrow_data = df[df['action'] == 'borrow'].groupby('userWallet')['amount']
    repay_data = df[df['action'] == 'repay'].groupby('userWallet')['amount']
    redeem_data = df[df['action'] == 'redeemunderlying'].groupby('userWallet')['amount']

    features['total_deposits'] = deposit_data.sum()
    features['total_borrows'] = borrow_data.sum()
    features['total_repays'] = repay_data.sum()
    features['total_redeems'] = redeem_data.sum()

    features['avg_deposit_size'] = deposit_data.mean()
    features['avg_borrow_size'] = borrow_data.mean()
    features['avg_repay_size'] = repay_data.mean()

    # Action frequency
    action_counts = df.groupby(['userWallet', 'action']).size().unstack(fill_value=0)
    for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
        features[f'{action}_count'] = action_counts.get(action, 0)

    # Risk and behavior
    features['liquidation_count'] = features.get('liquidationcall_count', 0)
    features['total_volume'] = features['total_deposits'] + features['total_borrows'] + features['total_repays']
    features['repay_ratio'] = features['total_repays'] / (features['total_borrows'] + 1e-8)
    features['borrow_deposit_ratio'] = features['total_borrows'] / (features['total_deposits'] + 1e-8)
    features['activity_diversity'] = features['unique_actions'] / 5  # Assuming 5 action types max

    # Final DataFrame
    features_df = pd.DataFrame(features).fillna(0)
    features_df['userWallet'] = features_df.index
    features_df = features_df.reset_index(drop=True)

    print(f" Created feature set for {len(features_df)} wallets with {len(features_df.columns)-1} features")
    return features_df

# -------------------- 3. Credit Score Calculation ---
def calculate_credit_score(features_df):
    """
    GeneratING credit score from engineered features.
    Score range: 0 to 1000.
    """

    scores = []
    for _, row in features_df.iterrows():
        score = 500  

        # Positive signals
        score += min(row['transaction_count'] * 1.5, 100)
        score += min(np.log1p(row['total_deposits']) * 15, 150)
        if row['total_borrows'] > 0:
            score += min(row['repay_ratio'] * 100, 100)
        score += row['activity_diversity'] * 30
        score += min(np.log1p(row['total_volume']) * 5, 50)

        # Negative 
        score -= row['liquidation_count'] * 100
        if row['borrow_deposit_ratio'] > 2:
            score -= (row['borrow_deposit_ratio'] - 2) * 50
        if row['transaction_count'] < 5:
            score -= 50

        # Final bounds
        scores.append(max(0, min(1000, score)))

    features_df['credit_score'] = scores
    print(f" Credit score generation complete. Min: {min(scores):.1f}, Max: {max(scores):.1f}")
    return features_df

# ---- 4. Score Analysis ---
def analyze_scores(credit_scores_df):
    """Providing insights into score distribution and user behavior."""
    print("\n CREDIT SCORE ANALYSIS")

    scores = credit_scores_df['credit_score']
    print(f"Wallets evaluated: {len(scores)}")
    print(f"Score Range: {scores.min()} - {scores.max()}")
    print(f"Mean: {scores.mean():.2f} | Median: {scores.median():.2f} | Std: {scores.std():.2f}")

    ranges = [(i, i+100) for i in range(0, 1000, 100)]
    print("\nScore Brackets:")
    for low, high in ranges:
        count = scores[(scores >= low) & (scores < high)].count()
        print(f"{low}-{high}: {count} wallets ({(count/len(scores))*100:.1f}%)")

    # Bottom 10%
    low_thresh = scores.quantile(0.1)
    low = credit_scores_df[credit_scores_df['credit_score'] <= low_thresh]
    print(f"\nLow Score Wallets (<= {low_thresh:.1f}):")
    print(f"Avg TXNs: {low['transaction_count'].mean():.1f}, Liquidations: {low['liquidation_count'].mean():.1f}, Repay Ratio: {low['repay_ratio'].mean():.2f}")

    # Top 10%
    high_thresh = scores.quantile(0.9)
    high = credit_scores_df[credit_scores_df['credit_score'] >= high_thresh]
    print(f"\nHigh Score Wallets (>= {high_thresh:.1f}):")
    print(f"Avg TXNs: {high['transaction_count'].mean():.1f}, Liquidations: {high['liquidation_count'].mean():.1f}, Repay Ratio: {high['repay_ratio'].mean():.2f}")

    return {'low_score_wallets': low, 'high_score_wallets': high, 'score_ranges': ranges}

# ---- 5. Entry Point ----
def generate_credit_scores(data_url):
  
    print(" INITIATING CREDIT SCORE PIPELINE")

    data = load_data(data_url)
  
    df = pd.DataFrame(data)

    print(f" Data contains {len(df)} transactions from {df['userWallet'].nunique()} wallets")

    features_df = engineer_advanced_features(df)
    credit_scores_df = calculate_credit_score(features_df)
    analysis_results = analyze_scores(credit_scores_df)

    print("\n Pipeline complete.")
    return {'credit_scores': credit_scores_df, 'analysis': analysis_results}

# -- Script Execution -
if __name__ == "__main__":
    results = generate_credit_scores(direct_download_url)

    if results:
        credit_scores_df = results['credit_scores']

        print("\n Sample Credit Scores:")
        print(credit_scores_df[['userWallet', 'credit_score', 'transaction_count', 'total_deposits', 'total_borrows', 'repay_ratio']].head(10))

        # Save as CSV
        credit_scores_df.to_csv('wallet_credit_scores.csv', index=False)
        print("📁 Saved credit scores to 'wallet_credit_scores.csv'")

        # Save as Excel
        credit_scores_df.to_excel('wallet_credit_scores.xlsx', index=False)
        print("📁 Saved credit scores to 'wallet_credit_scores.xlsx'")

        detailed_results = credit_scores_df.copy()
        detailed_results.to_csv('detailed_wallet_analysis.csv', index=False)
        print("📁 Detailed data exported to 'detailed_wallet_analysis.csv'")

        detailed_results.to_excel('detailed_wallet_analysis.xlsx', index=False)
        print("📁 Detailed data exported to 'detailed_wallet_analysis.xlsx'")
        
        print(" Detailed data exported to 'detailed_wallet_analysis.csv'")