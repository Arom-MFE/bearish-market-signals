"""
Daily SPY/KRE Put/Call Ratio Tracker (Calculates current PCR)
- Fetches option chain data for SPY and KRE from Yahoo Finance.
- Computes total put/call Volume_PCR and OI_PCR across expiries.
- Appends daily results (date, symbol, basis, puts, calls, PCR) to CSV.
- Skips appending if today's data already exists.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import os


# Core PCR computation
def pcr_for_symbol(symbol: str, basis: str = "volume", max_expiries: int | None = None) -> dict:
    """
    Compute aggregate Put/Call Ratio for a symbol across expiries.
    Parameters:
        symbol: ticker (e.g., 'SPY', 'KRE')
        basis:  'volume' or 'openInterest'
        max_expiries: optional limit on number of expiries
    Returns:
        dict containing total puts, calls, PCR, and expiries used
    """
    assert basis in ("volume", "openInterest")
    tk = yf.Ticker(symbol)
    expiries = tk.options or []
    if not expiries:
        raise RuntimeError(f"No option expiries found for {symbol}")
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    tot_puts = 0.0
    tot_calls = 0.0
    used = 0

    for exp in expiries:
        try:
            ch = tk.option_chain(exp)
            calls, puts = ch.calls, ch.puts
            c = float(calls[basis].fillna(0).sum())
            p = float(puts[basis].fillna(0).sum())
            tot_calls += c
            tot_puts += p
            used += 1
        except Exception:
            continue  # skip missing or illiquid expiries

    pcr = (tot_puts / tot_calls) if tot_calls > 0 else np.nan
    return {
        "Symbol": symbol,
        "Basis": basis.capitalize(),
        "Puts": int(round(tot_puts)),
        "Calls": int(round(tot_calls)),
        "PCR": round(float(pcr), 4) if np.isfinite(pcr) else None,
        "Expiries Used": used,
    }



# Daily snapshot and recording
def record_daily_pcr(csv_path: str = "pcr_history.csv") -> None:
    """
    Compute daily PCRs for SPY & KRE (volume + OI) and append to CSV.
    Skips if today's data already exists.
    """
    today = date.today().isoformat()
    print(f"\nRunning PCR snapshot for {today}...\n")

    symbols = ["SPY", "KRE"]
    bases = ["volume", "openInterest"]
    rows = []

    for sym in symbols:
        for b in bases:
            data = pcr_for_symbol(sym, basis=b)
            data["Date"] = today
            rows.append(data)

    df_today = pd.DataFrame(rows)[
        ["Date", "Symbol", "Basis", "Puts", "Calls", "PCR", "Expiries Used"]
    ]

    # Load existing data (if any)
    if os.path.exists(csv_path):
        df_hist = pd.read_csv(csv_path)
        if today in df_hist["Date"].unique():
            print("Today's PCR data already recorded — skipping append.\n")
            print(df_today.to_string(index=False))
            return
        df_hist = pd.concat([df_hist, df_today], ignore_index=True)
    else:
        df_hist = df_today

    df_hist.to_csv(csv_path, index=False)
    print(f"Appended today's PCR data to {csv_path}\n")
    print(df_today.to_string(index=False))


def main():
    record_daily_pcr("pcr_history.csv")


if __name__ == "__main__":
    main()
