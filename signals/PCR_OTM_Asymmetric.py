"""
Asymmetric DeepPCR for SPY & KRE — per-expiry OI/Volume PCR with plots.
Definition:
- Puts:  strike < S * (1 - δ)
- Calls: strike > S   (ALL OTM calls)
This asymmetric setup highlights demand for deep downside protection versus the full upside OTM book.
Outputs:
- Full per-expiry table with Asym_Volume_PCR and Asym_OI_PCR
- Top-N expiries by Asym_OI_PCR
- Bar plot of Asym_OI_PCR by expiry
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# Get spot price data 
def get_spot_price(symbol: str) -> float:
    """Get the latest spot price for the underlying."""
    tk = yf.Ticker(symbol)
    try:
        spot = float(getattr(tk, "fast_info", {}).get("last_price", np.nan))
    except Exception:
        spot = np.nan
    if not np.isfinite(spot) or spot <= 0:
        hist = tk.history(period="1d", auto_adjust=False)
        if hist.empty:
            raise RuntimeError(f"Could not fetch spot for {symbol}")
        spot = float(hist["Close"].iloc[-1])
    return spot


# Core calculation (Asymmetric DeepPCR)
def per_expiry_asym_deeppcr(
    symbol: str,
    mny_buffer: float,
    max_expiries: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each expiry:
      - Sum puts with strike < spot*(1 - mny_buffer)
      - Sum ALL OTM calls (strike > spot)
      - Compute Volume_PCR and OI_PCR for this asymmetric definition.
    """
    tk = yf.Ticker(symbol)
    expiries: List[str] = tk.options or []
    if not expiries:
        raise RuntimeError(f"No option expirations for {symbol}")
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    spot = get_spot_price(symbol)
    put_cut = spot * (1.0 - mny_buffer)
    call_cut = spot  # ALL OTM calls (any strike > spot)

    rows: List[Dict] = []
    for exp in expiries:
        try:
            ch = tk.option_chain(exp)
            calls = ch.calls.copy()
            puts = ch.puts.copy()

            # Ensure numeric types
            for col in ("strike", "volume", "openInterest"):
                if col in calls:
                    calls[col] = pd.to_numeric(calls[col], errors="coerce")
                if col in puts:
                    puts[col] = pd.to_numeric(puts[col], errors="coerce")

            otm_puts = puts.loc[puts["strike"] < put_cut]
            otm_calls = calls.loc[calls["strike"] > call_cut]

            p_vol = float(otm_puts["volume"].fillna(0).sum())
            c_vol = float(otm_calls["volume"].fillna(0).sum())
            p_oi = float(otm_puts["openInterest"].fillna(0).sum())
            c_oi = float(otm_calls["openInterest"].fillna(0).sum())

            vol_pcr = (p_vol / c_vol) if c_vol > 0 else np.nan
            oi_pcr = (p_oi / c_oi) if c_oi > 0 else np.nan

            rows.append(
                {
                    "Expiry": exp,
                    "Spot": round(spot, 2),
                    "Put_Volume": int(p_vol),
                    "Call_Volume": int(c_vol),
                    "Asym_Volume_PCR": None if np.isnan(vol_pcr) else float(vol_pcr),
                    "Put_OI": int(p_oi),
                    "Call_OI": int(c_oi),
                    "Asym_OI_PCR": None if np.isnan(oi_pcr) else float(oi_pcr),
                    "Total_OI": int(p_oi + c_oi),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows).sort_values("Expiry").reset_index(drop=True)
    return df

# Display + Plot
def show_asym_deeppcr_tables(
    symbol: str,
    thresholds: List[float],
    top_n: int = 5,
    max_expiries: Optional[int] = None,
) -> None:
    """Print per-expiry and top-N tables for Asymmetric DeepPCR*, with OI_PCR plots."""
    print(f"\n{'='*90}\n{symbol}: Asymmetric DeepPCR* (Puts < S*(1-δ) vs ALL OTM Calls)\n{'='*90}")

    for thr in thresholds:
        df = per_expiry_asym_deeppcr(symbol, mny_buffer=thr / 100.0, max_expiries=max_expiries)
        if df.empty:
            continue

        print(f"\n--- {symbol} | Threshold: {thr:.1f}% below spot ---")
        print(
            df[
                [
                    "Expiry",
                    "Spot",
                    "Put_Volume",
                    "Call_Volume",
                    "Asym_Volume_PCR",
                    "Put_OI",
                    "Call_OI",
                    "Asym_OI_PCR",
                    "Total_OI",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        # Top-N table
        top = (
            df[df["Asym_OI_PCR"].notna()]
            .sort_values("Asym_OI_PCR", ascending=False)
            .head(top_n)[
                ["Expiry", "Asym_OI_PCR", "Asym_Volume_PCR", "Put_OI", "Call_OI", "Total_OI", "Spot"]
            ]
        )
        print(
            f"\nTop {top_n} expiries by Asymmetric OI_PCR "
            f"(deep puts dominate most at {thr:.1f}% threshold):"
        )
        print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.bar(df["Expiry"], df["Asym_OI_PCR"], color="darkorange", alpha=0.85)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{symbol}: Asymmetric DeepPCR OI_PCR by Expiry ({thr:.1f}% below spot)")
        plt.xlabel("Expiration Date")
        plt.ylabel("Asym_OI_PCR (Puts < S*(1-δ) vs All OTM Calls)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

def main():
    MAX_EXPIRIES = None  # set an int (e.g., 6) to speed up queries
    TOP_N = 5
    THRESHOLDS = [2.5, 5.0, 10.0, 20.0]  # % below spot

    for sym in ["SPY", "KRE"]:
        show_asym_deeppcr_tables(sym, thresholds=THRESHOLDS, top_n=TOP_N, max_expiries=MAX_EXPIRIES)


if __name__ == "__main__":
    main()
