"""
Symmetric OTM/DeepPCR per expiry for SPY & KRE.
  * Standard OTM_PCR  (δ = 0%) → puts: strike < S, calls: strike > S
  * Deep, symmetric PCR (δ > 0%) → puts: strike < S*(1-δ), calls: strike > S*(1+δ)
- Computes per-expiry Volume_PCR and OI_PCR for each δ in a list
- Prints full table and Top-N by OI_PCR
- Plots OI_PCR by expiry for each δ
"""

from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Get spot price
def get_spot_price(symbol: str) -> float:
    """Get latest spot (last price or last close) for the underlying."""
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

# Computation PCR for different OTM threshold
def compute_symmetric_pcr_per_expiry(
    symbol: str,
    mny_buffer: float = 0.0,
    max_expiries: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each expiry:
      - Puts:  strike < spot * (1 - mny_buffer)
      - Calls: strike > spot * (1 + mny_buffer)
      - Sum put/call Volume & Open Interest
      - Compute Volume_PCR and OI_PCR (symmetric window)
    Returns: one row per expiry.
    """
    tk = yf.Ticker(symbol)
    expiries: List[str] = tk.options or []
    if not expiries:
        raise RuntimeError(f"No option expirations for {symbol}")
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    spot = get_spot_price(symbol)
    put_cut = spot * (1.0 - mny_buffer)
    call_cut = spot * (1.0 + mny_buffer)

    rows: List[Dict] = []
    for exp in expiries:
        try:
            ch = tk.option_chain(exp)
            calls = ch.calls.copy()
            puts = ch.puts.copy()

            # Ensure numeric columns
            for col in ("strike", "volume", "openInterest"):
                if col in calls:
                    calls[col] = pd.to_numeric(calls[col], errors="coerce")
                if col in puts:
                    puts[col] = pd.to_numeric(puts[col], errors="coerce")

            # Symmetric selection around spot
            sel_puts = puts.loc[puts["strike"] < put_cut]
            sel_calls = calls.loc[calls["strike"] > call_cut]

            p_vol = float(sel_puts["volume"].fillna(0).sum())
            c_vol = float(sel_calls["volume"].fillna(0).sum())
            p_oi = float(sel_puts["openInterest"].fillna(0).sum())
            c_oi = float(sel_calls["openInterest"].fillna(0).sum())

            vol_pcr = (p_vol / c_vol) if c_vol > 0 else np.nan
            oi_pcr = (p_oi / c_oi) if c_oi > 0 else np.nan

            rows.append(
                {
                    "Expiry": exp,
                    "Spot": round(spot, 4),
                    "Put_Volume": int(round(p_vol)),
                    "Call_Volume": int(round(c_vol)),
                    "Volume_PCR": None if np.isnan(vol_pcr) else float(vol_pcr),
                    "Put_OI": int(round(p_oi)),
                    "Call_OI": int(round(c_oi)),
                    "OI_PCR": None if np.isnan(oi_pcr) else float(oi_pcr),
                    "Total_OI": int(round(p_oi + c_oi)),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows).sort_values("Expiry").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No chain data aggregated for {symbol}")
    return df

# Display & plot
def show_symmetric_pcr_tables_and_plots(
    symbol: str,
    thresholds: List[float],             # e.g., [0.0, 0.025, 0.05, 0.10]
    top_n: int = 5,
    max_expiries: Optional[int] = None,
    min_total_oi: Optional[int] = None, # optional filter for thin expiries
) -> None:
    """
    For each δ in thresholds (0.0 = standard OTM):
      - Print full per-expiry table
      - Print Top-N expiries by OI_PCR
      - Plot OI_PCR by expiry
    """
    print(f"\n{'='*90}\n{symbol}: Symmetric PCR per Expiry (δ in {', '.join([str(int(t*100))+'%' for t in thresholds])})\n{'='*90}")

    for delta in thresholds:
        df = compute_symmetric_pcr_per_expiry(symbol, mny_buffer=delta, max_expiries=max_expiries)

        if min_total_oi is not None:
            df = df[df["Total_OI"] >= int(min_total_oi)].reset_index(drop=True)

        label = f"{int(delta*100)}%" if delta > 0 else "0% (Standard OTM)"
        print(f"\n--- {symbol} | Threshold δ = {label} ---")
        print(
            df[
                [
                    "Expiry",
                    "Spot",
                    "Put_Volume",
                    "Call_Volume",
                    "Volume_PCR",
                    "Put_OI",
                    "Call_OI",
                    "OI_PCR",
                    "Total_OI",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )

        top = (
            df[df["OI_PCR"].notna()]
            .sort_values("OI_PCR", ascending=False)
            .head(top_n)[["Expiry", "OI_PCR", "Volume_PCR", "Put_OI", "Call_OI", "Total_OI", "Spot"]]
        )
        print(f"\nTop {top_n} expiries by OI_PCR (δ = {label})")
        print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        # Plot OI_PCR by Expiry for this δ
        plt.figure(figsize=(10, 5))
        plt.bar(df["Expiry"], df["OI_PCR"], alpha=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{symbol}: OI_PCR per Expiry (δ = {label})")
        plt.xlabel("Expiration Date")
        plt.ylabel("OI_PCR")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()


# Demo for SPY and KRE
def main():
    MAX_EXPIRIES = None
    TOP_N = 5
    THRESHOLDS = [0.0, 0.025, 0.05, 0.10, 0.20]  # include 0.0 for standard OTM_PCR
    MIN_TOTAL_OI = None

    for sym in ["SPY", "KRE"]:
        show_symmetric_pcr_tables_and_plots(
            sym,
            thresholds=THRESHOLDS,
            top_n=TOP_N,
            max_expiries=MAX_EXPIRIES,
            min_total_oi=MIN_TOTAL_OI,
        )


if __name__ == "__main__":
    main()
