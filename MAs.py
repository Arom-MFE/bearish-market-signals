"""
This script downloads SPY and KRE price data, calculates their
50-day and 200-day moving averages,
plots both with clear trend visuals, and prints the latest market condition interpretation
(uptrend, correction, downtrend, or recovery) based on MA crossovers.
"""
import matplotlib.pyplot as plt
import yfinance as yf

# 1) Condition function
def stock_condition(price: float, ma50: float, ma200: float) -> tuple[str, str, str]:
    """
    Return (condition, meaning, action) based on price vs MA50 vs MA200.
    """
    if price > ma50 > ma200:
        return (
            "Price > MA50 > MA200",
            "Strong uptrend (momentum healthy)",
            "Wait for weakness; hedges likely underperform now.",
        )
    if price < ma50 and price > ma200:
        return (
            "Price < MA50 but > MA200",
            "Short-term correction inside long-term uptrend",
            "Watch for breakdown confirmation.",
        )
    if ma50 < ma200:
        return (
            "MA50 < MA200 (Death Cross)",
            "Downtrend confirmed",
            "Consider/hold recession hedges (e.g., index puts).",
        )
    if ma50 > ma200:
        return (
            "MA50 > MA200 (Golden Cross)",
            "Trend recovery",
            "Reduce hedges; re-risk mode.",
        )
    return ("Unclassified", "Mixed signals", "No action")


# 2) Compute MAs
def compute_ma(ticker: str, period: str = "2y"):
    """
    Download adjusted prices and compute MA50/MA200.
    Returns a DataFrame with columns: Close, MA50, MA200.
    """
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No data returned for {ticker}.")
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=200).mean()
    return df


# 3) Plot helper
def plot_ma(df, ticker: str):
    """
    Plot Close, MA50, MA200.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.2)
    plt.plot(df.index, df["MA50"], label="50-Day MA", linestyle="--")
    plt.plot(df.index, df["MA200"], label="200-Day MA", linestyle=":")
    plt.title(f"{ticker} — 50 & 200 Day Moving Averages")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 4) Print summary helper
def print_summary(df, ticker: str):
    """
    Print latest valid (non-NaN) values and the interpreted condition.
    """
    valid = df[df["MA50"].notna() & df["MA200"].notna()]
    if valid.empty:
        print(f"{ticker}: Not enough history for both MAs (need at least 200 trading days).")
        return
    row = valid.iloc[-1]
    price = float(valid["Close"].iloc[-1])
    ma50 = float(valid["MA50"].iloc[-1])
    ma200 = float(valid["MA200"].iloc[-1])

    cond, meaning, action = stock_condition(price, ma50, ma200)

    print(f"{ticker} latest")
    print(f"  Price: {price:.2f} | MA50: {ma50:.2f} | MA200: {ma200:.2f}")
    print(f"  Condition: {cond}")
    print(f"  Meaning:   {meaning}")
    print(f"  Action:    {action}")
    print()


def main():
    # Demo for SPY and KRE
    spy = compute_ma("SPY", period="2y")
    kre = compute_ma("KRE", period="2y")

    plot_ma(spy, "SPY")
    plot_ma(kre, "KRE")

    print_summary(spy, "SPY")
    print_summary(kre, "KRE")

if __name__ == "__main__":
    main()
