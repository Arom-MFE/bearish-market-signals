"""
This script downloads option chain data for SPY and KRE from Yahoo Finance,
aggregates total put/call volume and open interest for each expiry,
calculates per-expiry Put/Call Ratios (Volume_PCR and OI_PCR),
prints tables of results, plots OI_PCR by expiry, and appends daily data to CSV.
"""
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Core data collection
def per_expiry_totals(symbol: str, max_expiries: Optional[int] = None) -> pd.DataFrame:
    """Sum put/call Volume & OI for each expiry and compute PCRs."""
    tk = yf.Ticker(symbol)
    expiries: List[str] = tk.options or []
    if not expiries:
        raise RuntimeError(f"No option expirations for {symbol}")
    if max_expiries:
        expiries = expiries[:max_expiries]

    rows: List[Dict] = []
    for exp in expiries:
        try:
            ch = tk.option_chain(exp)
            calls, puts = ch.calls, ch.puts
            c_vol, p_vol = calls["volume"].fillna(0).sum(), puts["volume"].fillna(0).sum()
            c_oi,  p_oi  = calls["openInterest"].fillna(0).sum(), puts["openInterest"].fillna(0).sum()

            vol_pcr = p_vol / c_vol if c_vol > 0 else np.nan
            oi_pcr  = p_oi  / c_oi  if c_oi  > 0 else np.nan
            rows.append({
                "Expiry": exp,
                "Put_Volume": int(round(p_vol)),
                "Call_Volume": int(round(c_vol)),
                "Volume_PCR": None if np.isnan(vol_pcr) else float(vol_pcr),
                "Put_OI": int(round(p_oi)),
                "Call_OI": int(round(c_oi)),
                "OI_PCR": None if np.isnan(oi_pcr) else float(oi_pcr),
                "Total_OI": int(round(p_oi + c_oi)),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("Expiry").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No chain data aggregated for {symbol}")
    return df


# Display, save & plot
def show_top_by_oi_pcr(symbol: str, max_expiries: Optional[int] = None, top_n: int = 5, save_path="pcr_day__history.csv"):
    """Display full per-expiry table, top expiries by OI_PCR, and plot distribution."""
    df = per_expiry_totals(symbol, max_expiries=max_expiries)
    today = datetime.now().strftime("%Y-%m-%d")

    # Add Date and Symbol columns
    df["Date"] = today
    df["Symbol"] = symbol

    # Save / Append to CSV
    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)
        # Avoid duplicates for same symbol and date
        existing = existing[~((existing["Date"] == today) & (existing["Symbol"] == symbol))]
        df_all = pd.concat([existing, df], ignore_index=True)
    else:
        df_all = df

    df_all.to_csv(save_path, index=False)
    print(f"\n Saved updated data for {symbol} to '{save_path}' ({len(df)} rows added).")

    # --- Display Tables
    print(f"\n=== {symbol}: Per-expiry totals ===")
    print(
        df[["Expiry","Put_Volume","Call_Volume","Volume_PCR",
            "Put_OI","Call_OI","OI_PCR","Total_OI"]]
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print(f"\n{symbol}: Top {top_n} expiries by **OI_PCR** (puts dominate calls most)")
    top = (
        df[df["OI_PCR"].notna()]
        .sort_values("OI_PCR", ascending=False)
        .head(top_n)[["Expiry","OI_PCR","Volume_PCR","Put_OI","Call_OI","Total_OI"]]
    )
    print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- Plot OI_PCR by Expiry
    plt.figure(figsize=(10, 5))
    plt.bar(df["Expiry"], df["OI_PCR"], color="darkred", alpha=0.75)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{symbol}: OI_PCR per Expiry (Put/Call Open Interest Ratio)")
    plt.xlabel("Expiration Date")
    plt.ylabel("OI_PCR")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def main():
    MAX = None  # limit expiries if needed for speed
    for sym in ["SPY", "KRE"]:
        show_top_by_oi_pcr(sym, max_expiries=MAX, top_n=5)


if __name__ == "__main__":
    main()
