import sys
import os

from dotenv import load_dotenv
import pandas as pd

# Load API key from .env at project root
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Add src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from data_fetcher import fetch_news, fetch_stock_data_yfinance
from data_cleaner import clean_data, format_date_column
from sentiment_analysis import analyze_sentiment, aggregate_sentiments_by_date
from stock_returns import calculate_daily_returns
from data_merger import merge_data, merge_sentiment_with_stock
from correlation_analysis import calculate_correlation
from visualizationt import plot_daily_returns, scatter_plot

# Ticker symbol → company name
TICKERS = {
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'META': 'Meta',
    'MSFT': 'Microsoft',
    'NVDA': 'Nvidia',
    'TSLA': 'Tesla',
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
NEWS_CACHE = os.path.join(DATA_DIR, 'news_cache.csv')
STOCK_CACHE = os.path.join(DATA_DIR, 'stock_cache.csv')


def main():
    if not API_KEY:
        raise EnvironmentError("ALPHA_VANTAGE_API_KEY not found. Check your .env file.")

    os.makedirs(DATA_DIR, exist_ok=True)

    # --- News data (use cache if available to preserve API quota) ---
    if os.path.exists(NEWS_CACHE):
        print(f"Loading news data from cache: {NEWS_CACHE}")
        news_df = pd.read_csv(NEWS_CACHE)
    else:
        print("Fetching news data from Alpha Vantage...")
        # stock_df must be fetched first so we can pass its dates as fallback
        if not os.path.exists(STOCK_CACHE):
            print("Fetching stock data via yfinance first (needed for fallback dates)...")
            _tmp_stock = fetch_stock_data_yfinance(TICKERS, period='6mo')
            _tmp_stock.to_csv(STOCK_CACHE, index=False)
        else:
            _tmp_stock = pd.read_csv(STOCK_CACHE)
        fallback_dates = sorted(_tmp_stock['date'].dropna().unique().tolist())
        news_df = fetch_news(list(TICKERS.keys()), API_KEY,
                             tickers_map=TICKERS, fallback_dates=fallback_dates)
        news_df.to_csv(NEWS_CACHE, index=False)
        print(f"  Saved to {NEWS_CACHE}")

    # --- Stock data via yfinance (no API key needed, no rate limits) ---
    if os.path.exists(STOCK_CACHE):
        print(f"Loading stock data from cache: {STOCK_CACHE}")
        stock_df = pd.read_csv(STOCK_CACHE)
    else:
        print("Fetching stock data via yfinance...")
        stock_df = fetch_stock_data_yfinance(TICKERS, period='6mo')
        stock_df.to_csv(STOCK_CACHE, index=False)
        print(f"  Saved to {STOCK_CACHE}")

    # Clean data (date parsing, drop NAs)
    stock_df = clean_data(stock_df, 'date')
    news_df = clean_data(news_df, 'date')

    # Format the 'date' columns to YYYY-MM-DD strings
    news_df = format_date_column(news_df, 'date')
    stock_df = format_date_column(stock_df, 'date')

    # Merge news with stock data on date
    merged_df = merge_data(news_df, stock_df)

    # Analyze sentiment on headlines
    merged_df['sentiment'] = merged_df['headline'].apply(analyze_sentiment)

    # Calculate daily stock returns
    merged_df = calculate_daily_returns(merged_df)

    # Aggregate sentiments by date
    daily_sentiment = aggregate_sentiments_by_date(merged_df)

    # Merge sentiment data with stock returns
    daily_df = merge_sentiment_with_stock(merged_df, daily_sentiment)

    # Calculate and print Pearson correlation
    correlation = calculate_correlation(daily_df)
    print(f'Pearson correlation coefficient: {correlation}')

    # Visualize
    plot_daily_returns(daily_df)
    scatter_plot(daily_df)


if __name__ == '__main__':
    main()