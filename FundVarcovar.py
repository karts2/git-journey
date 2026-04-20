import yfinance as yf
import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize

FUND_POOL = {
    # US Equity
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq-100 ETF",
    "IWM":  "Russell 2000 ETF",
    "VTI":  "Total US Market ETF",
    "VFIAX":"Vanguard 500 Index Fund",
    # International Equity
    "EFA":  "Developed Markets ETF",
    "EEM":  "Emerging Markets ETF",
    "VEU":  "All-World ex-US ETF",
    "VXUS": "Total Intl Stock ETF",
    # Fixed Income
    "AGG":  "US Aggregate Bond ETF",
    "BND":  "Total Bond Market ETF",
    "TLT":  "20+ Year Treasury ETF",
    "LQD":  "Investment Grade Corp Bond ETF",
    "HYG":  "High Yield Bond ETF",
    # Real Assets
    "VNQ":  "Real Estate ETF",
    "GLD":  "Gold ETF",
    "IAU":  "iShares Gold ETF",
    # Sector ETFs
    "XLK":  "Technology Sector ETF",
    "XLF":  "Financials Sector ETF",
    "XLV":  "Healthcare Sector ETF",
    "XLE":  "Energy Sector ETF",
    "XLU":  "Utilities Sector ETF",
    # Multi-asset / Balanced
    "AOM":  "Moderate Allocation ETF",
    "AOA":  "Aggressive Allocation ETF",
    "VBINX":"Vanguard Balanced Index Fund",
}

# Approximate ETF -> top holdings mapping (static, for converting ETF allocations
# into concrete stock picks). We use simple relative weights that sum to 1.0.
ETF_TO_HOLDINGS = {
    "SPY": {"AAPL": 0.06, "MSFT": 0.06, "AMZN": 0.04, "NVDA": 0.035, "GOOGL": 0.035, "BRK-B": 0.02, "JNJ": 0.02, "V": 0.02, "JPM": 0.02, "UNH": 0.02},
    "QQQ": {"AAPL": 0.15, "MSFT": 0.14, "NVDA": 0.12, "AMZN": 0.06, "TSLA": 0.05, "GOOGL": 0.05, "META": 0.05, "ADBE": 0.04, "PYPL": 0.03, "INTC": 0.03},
    "IWM": {"AMC": 0.03, "RBLX": 0.03, "SBUX": 0.03, "DOCU": 0.03, "MRO": 0.03, "UAL": 0.03, "KMX": 0.03, "NKE": 0.08, "TGT": 0.08, "F": 0.04},
    "VTI": {"AAPL": 0.05, "MSFT": 0.05, "AMZN": 0.03, "NVDA": 0.03, "GOOGL": 0.03, "V": 0.02, "JPM": 0.02, "JNJ": 0.02, "PG": 0.02, "HD": 0.02},
    "VFIAX": {"AAPL": 0.06, "MSFT": 0.06, "AMZN": 0.04, "NVDA": 0.035, "GOOGL": 0.035, "BRK-B": 0.02, "JNJ": 0.02, "V": 0.02, "JPM": 0.02, "UNH": 0.02},
    "EFA": {"NESN.SW": 0.04, "ASML": 0.04, "SHEL": 0.03, "SAP": 0.03, "RIO": 0.03, "OR.PA": 0.03, "HSBC": 0.03, "AZN": 0.03, "BP": 0.03, "SAP.DE": 0.03},
    "EEM": {"TSM": 0.08, "BABA": 0.06, "TCEHY": 0.06, "NIO": 0.02, "HDB": 0.02, "ICBC": 0.02, "PDD": 0.03, "JD": 0.03, "VALE": 0.03, "POSCO": 0.02},
    "VXUS": {"TSM": 0.05, "NVS": 0.04, "ASML": 0.04, "RDS.A": 0.03, "TM": 0.03, "SAP": 0.03, "SNY": 0.03, "NOVO-B": 0.03, "NESN.SW": 0.03, "BABA": 0.03},
    "VNQ": {"PLD": 0.12, "SPG": 0.08, "WELL": 0.06, "PSA": 0.06, "EQR": 0.05, "O": 0.05, "VTR": 0.05, "AVB": 0.05, "DLR": 0.05, "EQIX": 0.05},
    "XLK": {"MSFT": 0.18, "AAPL": 0.16, "NVDA": 0.12, "ADBE": 0.05, "AVGO": 0.05, "CRM": 0.04, "INTU": 0.04, "ORCL": 0.03, "CSCO": 0.03, "TXN": 0.03},
    "XLU": {"NEE": 0.1, "DUK": 0.09, "SO": 0.08, "AEP": 0.08, "EXC": 0.07, "SRE": 0.06, "PEG": 0.06, "D": 0.06, "ES": 0.06, "PPL": 0.06},
    "GLD": {"GLD": 1.0},
    "IAU": {"IAU": 1.0},
    "AGG": {"US_TREASURY": 0.6, "INV_GRADE_CORP": 0.4},
    "BND": {"US_TREASURY": 0.6, "INV_GRADE_CORP": 0.4},
    "TLT": {"US_TREASURY_LONG": 1.0},
    "LQD": {"INV_GRADE_CORP": 1.0},
    "HYG": {"HIGH_YIELD_CORP": 1.0},
}


def expand_etf_allocations(etf_weights: dict[str, float], n_stocks: int = 10) -> dict:
    """Expand ETF allocations into concrete stock-level weights and return top n_stocks.

    etf_weights: mapping ETF ticker -> percent (0-100)
    returns: dict of top n_stocks ticker -> percent (sum to ~100)
    """
    stock_w = {}
    for etf, pct in etf_weights.items():
        if pct == 0 or etf not in ETF_TO_HOLDINGS:
            continue
        holdings = ETF_TO_HOLDINGS[etf]
        for h, rel in holdings.items():
            stock_w[h] = stock_w.get(h, 0.0) + pct * rel / 100.0

    # normalize and pick top n_stocks
    if not stock_w:
        return {}
    total = sum(stock_w.values())
    for k in list(stock_w.keys()):
        stock_w[k] = round(stock_w[k] / total * 100.0, 2)

    # sort and take top n_stocks
    items = sorted(stock_w.items(), key=lambda x: x[1], reverse=True)[:n_stocks]
    return dict(items)


def convert_to_stock_picks(category_index: int = 3, n_stocks: int = 10) -> dict:
    """Helper: get ETF portfolio for category and convert to stock picks."""
    port = suggest_portfolio_for_category(category_index=category_index)
    if not port:
        return {}
    etf_weights = {k: v for k, v in port["weights_percent"].items()}
    return expand_etf_allocations(etf_weights, n_stocks=n_stocks)


def render_picks_html(picks: dict, out_path: str = "stock_picks.html", title: str | None = None) -> str:
        """Render a simple HTML page showing the stock picks and weights. Returns path."""
        if not picks:
                raise ValueError("No picks to render")
        title = title or "Suggested 10 Stock Picks"
        rows = "\n".join([
                f"<tr><td>{ticker}</td><td style=\"text-align:right\">{weight:.2f}%</td></tr>"
                for ticker, weight in picks.items()
        ])
        html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, Helvetica, sans-serif; padding:20px; color:#222 }}
        h1 {{ font-size:20px }}
        table {{ border-collapse: collapse; width:360px }}
        th, td {{ border:1px solid #ddd; padding:8px }}
        th {{ background:#f6f6f6; text-align:left }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <table>
        <thead><tr><th>Ticker</th><th>Weight</th></tr></thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""
        with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
        print(f"Saved stock picks HTML to: {out_path}")
        return out_path


#the function fetches N years of adjusted closing prices for a list of tickers, 
#then returns a DataFrame of their daily percentage returns — a standard input for 
#portfolio analysis, volatility calculations, or correlation matrices.

def _fetch_daily_returns(
    tickers: list[str],
    years: int = 10,
) -> pd.DataFrame:
    prices = yf.download(
        tickers,
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
    )["Close"]
    return prices.pct_change().dropna()

#the function randomly picks n funds from a pool, fetches their historical daily returns, 
#and gives back their annualised mean return — one value per fund. Each call will return 
#a different random sample unless you fix the random seed.

def get_mean_returns(
    n: int = 10,
    years: int = 10,
    pool: dict[str, str] | None = None,
) -> pd.Series:
    fund_pool = pool or FUND_POOL
    tickers   = random.sample(list(fund_pool.keys()), n)
    returns   = _fetch_daily_returns(tickers, years)
    print(returns)
    return (returns.mean() * 252).rename("mean_return")


def get_var_covar_matrix(
    mean_returns: pd.Series,
    years: int = 10,
) -> pd.DataFrame:
    tickers = mean_returns.index.tolist()
    returns = _fetch_daily_returns(tickers, years)
    print((returns.cov() * 252)[tickers].loc[tickers])
    return (returns.cov() * 252)[tickers].loc[tickers]




 #this finds the portfolio allocation that achieves the lowest possible volatility 
 #given the assets available — a foundational building block in robo-advisor portfolio 
 #construction before you layer on return targets or risk preferences.

def gmvp(cov: pd.DataFrame, allow_short: bool) -> dict:
    n  = len(cov)
    w0 = np.ones(n) / n
    bounds = (None, None) if allow_short else (0.0, None)
    result = minimize(
        lambda w: w @ cov.values @ w, w0, method="SLSQP",
        bounds=[bounds] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    w = result.x
    return {"weights": w, "return": float(w @ mean_ret.values), "sigma": float(np.sqrt(w @ cov.values @ w))}

#this function is the core of modern portfolio theory in code — it answers "for every possible level of risk, 
#what's the best return you can achieve?" The output feeds directly into visualising portfolio choices for a user in a robo-advisor.

def efficient_frontier(mean_ret, cov, allow_short, n_points=200):
    n      = len(mean_ret)
    bounds = (None, None) if allow_short else (0.0, None)
    mu_min = gmvp(cov, allow_short)["return"]
    mu_max = mean_ret.max() * (1.3 if allow_short else 1.0)
    targets = np.linspace(mu_min, mu_max, n_points)
    sigmas, mus, weights = [], [], []
    for mu_t in targets:
        res = minimize(
            lambda w: w @ cov.values @ w, np.ones(n)/n, method="SLSQP",
            bounds=[bounds]*n,
            constraints=[
                {"type":"eq","fun":lambda w: w.sum()-1},
                {"type":"eq","fun":lambda w: w @ mean_ret.values - mu_t},
            ],
        )
        if res.success:
            sigmas.append(np.sqrt(res.fun))
            mus.append(mu_t)
            weights.append(res.x)
    sigmas = np.array(sigmas)
    mus = np.array(mus)
    weights = np.array(weights)

    # Save results (sigma, mu and individual asset weights) to CSV
    cols = ["sigma", "mu"] + list(mean_ret.index)
    if sigmas.size:
        df = pd.DataFrame(np.hstack([
            sigmas.reshape(-1, 1),
            mus.reshape(-1, 1),
            weights
        ]), columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
    df.to_csv("efficient_frontier.csv", index=False)

    return sigmas, mus, weights


def plot_efficient_frontier(mean_ret, cov):
    fig, ax = plt.subplots(figsize=(10, 6))
    for allow_short, color, ls, label in [
        (True,  "#7F77DD", "-",  "With short sales"),
        (False, "#1D9E75", "--", "No short sales"),
    ]:
        sig, mu, _ = efficient_frontier(mean_ret, cov, allow_short)
        ax.plot(sig*100, mu*100, color=color, ls=ls, lw=2.2, label=label, zorder=3)
        g = gmvp(cov, allow_short)
        ax.scatter(g["sigma"]*100, g["return"]*100, marker="^", s=120,
                   color=color, edgecolors="white", linewidths=1, zorder=5,
                   label=f"GMVP ({'short' if allow_short else 'no short'}): σ={g['sigma']*100:.1f}% μ={g['return']*100:.1f}%")
    fund_sigs = np.sqrt(np.diag(cov.values))
    for i, ticker in enumerate(mean_ret.index):
        ax.scatter(fund_sigs[i]*100, mean_ret.iloc[i]*100, color="#BA7517", s=60,
                   zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(ticker, (fund_sigs[i]*100, mean_ret.iloc[i]*100),
                    textcoords="offset points", xytext=(6,4), fontsize=8, color="#5F5E5A")
    ax.set_xlabel("Risk — σ (annualised %)")
    ax.set_ylabel("Return (annualised %)")
    ax.set_title("Efficient Frontier — with and without short sales")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    out_path = "efficient_frontier.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved efficient frontier plot to: {out_path}")


def suggest_portfolios_from_csv(csv_path: str = "efficient_frontier.csv") -> dict:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"efficient frontier CSV not found: {csv_path}")
        return {}
    if df.empty:
        print("efficient_frontier.csv is empty")
        return {}

    tickers = [c for c in df.columns if c not in ["sigma", "mu"]]
    low = df.loc[df["sigma"].idxmin()]
    mid = df.iloc[len(df) // 2]
    high = df.loc[df["sigma"].idxmax()]

    def extract_weights(row):
        w = row[tickers].astype(float).values
        s = w.sum()
        if s == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        return dict(zip(tickers, (w * 100).round(2)))

    out = {
        "conservative": {"sigma": float(low["sigma"]), "mu": float(low["mu"]), "weights": extract_weights(low)},
        "moderate": {"sigma": float(mid["sigma"]), "mu": float(mid["mu"]), "weights": extract_weights(mid)},
        "aggressive": {"sigma": float(high["sigma"]), "mu": float(high["mu"]), "weights": extract_weights(high)},
    }
    print(json.dumps(out, indent=2))
    return out


def ask_risk_questions() -> int:
    """Ask the user a short questionnaire and return a category index 0..5.

    Categories (0..5): Very Conservative, Conservative, Moderately Conservative,
    Moderate, Aggressive, Very Aggressive
    """
    print("\nQuick questionnaire to determine your risk category.")
    # Each answer maps to a numeric score; we'll combine and bin into 6 categories
    def ask(prompt, options):
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        while True:
            a = input("Select option number: ").strip()
            if a.isdigit() and 1 <= int(a) <= len(options):
                return int(a) - 1
            print("Please enter a valid option number.")

    score = 0
    # Horizon
    score += ask("Investment horizon?", ["<3 years", "3-7 years", ">7 years"]) * 2
    # Risk tolerance
    score += ask("Risk tolerance?", ["Very low", "Low", "Moderate", "High"]) 
    # Maximum drawdown tolerance
    score += ask("Max drawdown you could tolerate without selling?", ["<=10%", "11-25%", "26-50%", ">50%"]) 
    # Goal
    score += ask("Primary goal?", ["Capital preservation", "Income", "Growth"]) 
    # Liquidity needs
    score += ask("Liquidity needs?", ["High (need cash soon)", "Medium", "Low (can lock for years)"]) 
    # Constraints
    c = ask("Any constraints?", ["None", "Prefer ETFs", "ESG / ethical screens"]) 
    score += c

    # Map raw score to 6 buckets
    min_score = 0
    max_score = (2*2) + 3 + 3 + 2 + 2 + 2  # approximate upper bound
    # normalize to 0..5
    bins = np.linspace(min_score, max_score, 6)
    # find which bin score falls into
    cat = int(np.digitize(score, bins, right=False))
    cat = max(0, min(5, cat - 1))
    names = [
        "Very Conservative",
        "Conservative",
        "Moderately Conservative",
        "Moderate",
        "Aggressive",
        "Very Aggressive",
    ]
    print(f"Selected category: {names[cat]} (score={score})")
    return cat


def suggest_portfolio_for_category(csv_path: str = "efficient_frontier.csv", category_index: int = 3) -> dict:
    """Pick a portfolio from the efficient frontier CSV corresponding to one of six risk categories.

    category_index: 0..5 (0=Very Conservative, 5=Very Aggressive)
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"efficient frontier CSV not found: {csv_path}")
        return {}
    if df.empty:
        print("efficient_frontier.csv is empty")
        return {}

    tickers = [c for c in df.columns if c not in ["sigma", "mu"]]
    sigmas = df["sigma"].values
    # map category to percentile grid 0..100
    percentiles = np.linspace(0, 100, 6)
    target = np.percentile(sigmas, percentiles[category_index])
    idx = int(np.abs(sigmas - target).argmin())
    row = df.iloc[idx]

    w = row[tickers].astype(float).values
    s = w.sum()
    if s == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s

    out = {
        "category_index": int(category_index),
        "category_name": [
            "Very Conservative",
            "Conservative",
            "Moderately Conservative",
            "Moderate",
            "Aggressive",
            "Very Aggressive",
        ][category_index],
        "sigma": float(row["sigma"]),
        "mu": float(row["mu"]),
        "weights_percent": dict(zip(tickers, (w * 100).round(2)))
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    mean_ret   = get_mean_returns()
    cov_matrix = get_var_covar_matrix(mean_ret)
    plot_efficient_frontier(mean_ret, cov_matrix)
    # Suggest portfolios using the CSV produced by efficient_frontier
    suggest_portfolios_from_csv()
    # Interactive questionnaire to classify into 6 categories and suggest portfolio
    try:
        cat = ask_risk_questions()
        suggest_portfolio_for_category(category_index=cat)
    except Exception as e:
        print(f"Questionnaire skipped or failed: {e}")
    # Convert ETF allocations for the selected category into 10 concrete stock picks
    try:
        picks = convert_to_stock_picks(category_index=cat, n_stocks=10)
        if picks:
            print("Suggested 10 stock picks:")
            print(json.dumps(picks, indent=2))
            try:
                html_path = render_picks_html(picks, out_path="stock_picks.html")
                print(f"HTML saved: {html_path}")
            except Exception as e:
                print(f"Failed to render HTML: {e}")
    except Exception as e:
        print(f"Failed to convert to stock picks: {e}")
