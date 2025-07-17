# Run Directly On Colab Notebook

You can run this project instantly in your browser using Google Colab:  
[Open in Colab](https://colab.research.google.com/drive/12DRLZMmh7YV8SdG0f270eTt5jHSfwTiU?usp=sharing)

---

# DeFi Credit Scoring System

A machine learning-based credit scoring system for DeFi wallets using Aave V2 transaction data.

---

## Getting Started (Local)

```bash
git clone https://github.com/Ankit-03G/Zeru.git
cd Zeru
pip install -r requirements.txt
# Or, if requirements.txt is missing:
pip install pandas numpy requests scikit-learn matplotlib seaborn
python credit_scoring_system.py
```

---

## What Is This?

This system analyzes DeFi transaction patterns to assign credit scores (0-1000) to wallet addresses based on their historical activity on the Aave V2 protocol.  
Higher scores mean more reliable usage; lower scores indicate riskier or exploitative behavior.

---

## Main Features

- Loads large-scale DeFi transaction data (Aave V2)
- Engineers wallet-level features: activity, financial, and risk indicators
- Calculates a credit score per wallet using domain knowledge
- Saves results and analysis to CSV/Excel files

---

## Process Flow

```
Raw JSON Data → Feature Engineering → Credit Score Calculation → Analysis & Visualization
```

---

## Key Engineered Features

- `transaction_count`, `unique_actions`, `days_active`, `avg_daily_transactions`
- `total_deposits`, `total_borrows`, `total_repays`, `total_redeems`
- `avg_deposit_size`, `avg_borrow_size`, `avg_repay_size`
- `liquidation_count`, `total_volume`, `repay_ratio`, `borrow_deposit_ratio`, `activity_diversity`
- `deposit_count`, `borrow_count`, `repay_count`, `redeemunderlying_count`, `liquidationcall_count`

---

## Credit Scoring Logic

- **Base score:** 500
- **Bonuses:** +1.5 per transaction (max 100), +15×log(deposits) (max 150), +100×repay_ratio (max 100), +30×diversity, +5×log(volume) (max 50)
- **Penalties:** -100 per liquidation, -50×(borrow_deposit_ratio-2) if ratio > 2, -50 if transaction_count < 5
- **Clamped to:** 0–1000

---

## Usage

1. Load data from Google Drive
2. Engineer features
3. Calculate credit scores
4. Output results/analysis to CSV and Excel

---

## Output Files

- `wallet_credit_scores.csv` and `.xlsx`: Credit scores per wallet
- `detailed_wallet_analysis.csv` and `.xlsx`: All features and scores

---

## Model Insights

On 100K transactions from 3,497 wallets:
- **Score range:** 0.0–924.0
- **Mean:** 662.5, **Median:** 615.0, **Std. Dev.:** 157.1

**Score Distribution:**
- 0-100: 2.4% (High-risk)
- 100-500: 4.0% (Medium-risk)
- 500-700: 64.5% (Standard)
- 700-900: 27.7% (High-quality)
- 900-1000: 4.3% (Premium)

---

## Typical Wallet Profiles

**Low-Score (Bottom 10%)**
- Few transactions (avg 18.8), low repay ratio (0.05), some liquidations
- Indicates risky/new users, poor repayment

**High-Score (Top 10%)**
- Many transactions (avg 96.5), high repay ratio (0.93), no liquidations
- Indicates active, reliable users

---

## Configuration

**Data source:**  
Google Drive file ID: `1ZiFWFr84rOn8WwU13MBsJUUQI6bIPEmL`  
Change in the script if using other sources.

**Scoring weights:**  
Tune them in `calculate_credit_score()` as needed.

---

## Extending

- Add new wallet features in `engineer_advanced_features()`
- Adjust scoring in `calculate_credit_score()`
- Try clustering or supervised learning for alternative scoring if you have labeled data
- Consider other DeFi protocols or time-series features

---

## Limitations

- No "ground truth" labels – scoring is domain-knowledge based
- Only Aave V2 is included (single protocol)
- Static weights; no learning from data
- No market condition or time-based adjustment (yet!)

---

## Future Plans

- Support more DeFi protocols
- Add time-series and market-based features
- Make scoring adaptive
- Enable real-time updates

---

## Requirements

- Python 3.7+
- pandas, numpy, requests, scikit-learn, matplotlib, seaborn

---

## License

MIT License

---

## Contributing

1. Fork this repo
2. Make a feature branch
3. Commit improvements and tests
4. Open a pull request

---
