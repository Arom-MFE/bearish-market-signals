# 🐻 bearish-market-signals

A collection of Python scripts for generating and analyzing **bearish market sentiment indicators**, focusing on **Put/Call Ratios (PCR)** and **moving average signals** for ETFs such as **SPY** and **KRE**.  
All data is pulled live from Yahoo Finance via the `yfinance` API.

---

### 📂 Project Structure

| Folder / File | Description |
|----------------|-------------|
| **signals/** | Contains all Python scripts for computing different bearish market indicators. |
| **signals/MAs.py** | Moving Average crossover strategy (50-day vs 200-day) to detect uptrend/downtrend shifts. |
| **signals/PCR_Per_Expiry.py** | Base Put/Call Ratio computed per expiry — shows put/call dominance across maturities. |
| **signals/PCR_OTM_Symmetric.py** | *Symmetric DeepPCR* — compares deep OTM puts *(S × (1 − δ))* vs deep OTM calls *(S × (1 + δ))* to capture balanced tail-risk hedging. |
| **signals/PCR_OTM_Asymmetric.py** | *Asymmetric DeepPCR* — compares deep OTM puts *(S × (1 − δ))* vs **all** OTM calls *(> S)* to measure downside hedging intensity. |
| **signals/PCR_Daily.py** | Tracks daily aggregated Put/Call Ratios for SPY & KRE and appends results to `pcr_history.csv`. |

