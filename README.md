# Quant-Research-Project-JaySalvi

## Overview
Volatility pairs trading strategy on Nifty and BankNifty implied volatility (10-min bars, 2021–2022).

## Data Preparation
- Raw data (`data.parquet`) contains minute-level gaps (overnight + occasional intraday)
- Imputation: Grouped linear interpolation **within each trading day only** (prevents look-ahead leakage)
- Filtered to official hours (09:15–15:30 IST) and resampled to clean 10-min bars
- Final dataset: 18,759 bars, zero missing values

## Base Model (Raw Spread Z-Score)
- As requested: no hedging, z-score on raw spread (BankNifty IV – Nifty IV)
- Result: Sharpe ~0.78, very few trades (structural drift makes spread non-stationary)

## Improved Model (Kalman Dynamic Hedging)
- Dynamic hedge ratio estimated using Kalman Filter
- Parameters derived from data (transition covariance = variance of beta changes) → regime-adaptive
- Z-score on hedged spread → truly mean-reverting series
- Realistic execution: TTE^0.7 vega scaling + 35 bps per IV point round-trip cost
- Result: Sharpe 2.19 (+180%), ~1,800 trades, max DD 0.72

## Key Insight
Raw spread is not tradable on medium frequency.  
Dynamic hedging via data-driven Kalman Filter unlocks strong, realistic alpha.

Full code, results, and inline plots in the Jupyter notebook.





— Jay Salvi, 
 November 2025
