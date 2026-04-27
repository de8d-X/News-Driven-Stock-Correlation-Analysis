import time
import random
import requests
import pandas as pd
import yfinance as yf

BASE_URL = "https://www.alphavantage.co/query"

# Sample financial headline templates used when Alpha Vantage quota is exhausted
_HEADLINE_TEMPLATES = [
    "{company} reports strong quarterly earnings, beating analyst expectations",
    "{company} stock rises after positive product launch announcement",
    "{company} faces regulatory scrutiny over data privacy concerns",
    "{company} announces share buyback program worth $5 billion",
    "{company} CEO outlines growth strategy at investor conference",
    "{company} revenue falls short of forecasts amid market slowdown",
    "{company} expands into new markets with strategic acquisition",
    "{company} cuts workforce as part of restructuring plan",
    "{company} partners with major firm to accelerate AI development",
    "{company} raises full-year guidance after strong performance",
    "Analysts upgrade {company} citing improving fundamentals",
    "{company} stock under pressure after disappointing guidance",
    "{company} launches new product line targeting enterprise customers",
    "Institutional investors increase stake in {company}",
    "{company} beats revenue estimates for the third consecutive quarter",
]


def _generate_sample_news(tickers_map: dict[str, str], dates: list[str]) -> pd.DataFrame:
    """Generate synthetic news headlines to allow a full offline demo run."""
    rows = []
    for date in dates:
        for ticker, company in tickers_map.items():
            n = random.randint(1, 3)
            for _ in range(n):
                headline = random.choice(_HEADLINE_TEMPLATES).format(company=company)
                rows.append({"headline": headline, "date": date, "stock": ticker})
    return pd.DataFrame(rows)


def fetch_news(tickers: list[str], api_key: str, tickers_map: dict[str, str] = None,
               fallback_dates: list[str] = None) -> pd.DataFrame:
    """Fetch news headlines from Alpha Vantage (per-ticker, free-tier safe).

    Falls back to synthetic headlines if the daily API quota is exhausted.
    Returns a DataFrame with columns: headline, date, stock.
    """
    all_rows = []
    quota_hit = False

    for i, ticker in enumerate(tickers):
        if quota_hit:
            break
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": 50,
            "apikey": api_key,
        }
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Information" in data or "Note" in data:
            print(f"  Alpha Vantage quota exhausted — switching to sample news data.")
            quota_hit = True
            break

        feed = data.get("feed", [])
        for article in feed:
            title = article.get("title", "")
            raw_time = article.get("time_published", "")
            date_raw = raw_time[:8]
            date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            all_rows.append({"headline": title, "date": date, "stock": ticker})

        print(f"  Fetched {len(feed)} news articles for {ticker}")

        if i < len(tickers) - 1:
            time.sleep(12)

    if quota_hit or not all_rows:
        if tickers_map and fallback_dates:
            print("  Generating sample news headlines for demo run...")
            return _generate_sample_news(tickers_map, fallback_dates)
        raise ValueError("No news data available and no fallback dates provided.")

    return pd.DataFrame(all_rows)


def fetch_stock_data_yfinance(tickers_map: dict[str, str], period: str = "6mo") -> pd.DataFrame:
    """Fetch daily OHLCV stock data using yfinance (no API key required).

    Args:
        tickers_map: dict mapping ticker → company name
        period: yfinance period string e.g. '6mo', '1y', '2y'
    Returns:
        Combined DataFrame with columns: date, Open, High, Low, Close, Volume, Company
    """
    frames = []
    for ticker, company in tickers_map.items():
        print(f"  Fetching stock data: {company} ({ticker}) via yfinance...")
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  Warning: no data returned for {ticker}, skipping.")
            continue
        raw = raw.reset_index()
        # yfinance may return MultiIndex columns — flatten them
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["Company"] = company
        frames.append(df)

    if not frames:
        raise ValueError("yfinance returned no data for any ticker.")

    return pd.concat(frames, ignore_index=True)

