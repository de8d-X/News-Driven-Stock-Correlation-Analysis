# Financial News Sentiment and Stock Market Correlation Analysis

This project analyzes whether the tone of financial news headlines is related to short-term stock market movement for major technology companies. It combines news sentiment scoring, stock return calculation, date-based data alignment, Pearson correlation analysis, and visualization in both script and notebook workflows.

## Overview

The current pipeline uses:

- Alpha Vantage for company news headlines
- yfinance for stock price history
- TextBlob for headline sentiment polarity
- pandas for cleaning and merging data
- matplotlib and seaborn for visualization

The analysis covers these companies:

- Apple (`AAPL`)
- Amazon (`AMZN`)
- Google (`GOOGL`)
- Meta (`META`)
- Microsoft (`MSFT`)
- Nvidia (`NVDA`)
- Tesla (`TSLA`)

## What The Project Does

The workflow is:

1. Fetch or load cached news headlines for each company.
2. Fetch or load cached daily stock prices.
3. Clean and normalize the datasets.
4. Align news and stock data by date.
5. Score each headline with sentiment polarity.
6. Compute daily stock returns.
7. Aggregate sentiment by day.
8. Calculate Pearson correlation between average daily sentiment and daily return.
9. Plot overall and per-company relationships.

## Current Data Behavior

News data is fetched from Alpha Vantage. Because the free tier has a strict daily quota, the project includes a fallback mode:

- If cached news data exists, it is reused.
- If the API quota is exceeded and no cache exists, the project generates sample news headlines so the pipeline can still run end-to-end.

Stock data is fetched from yfinance and cached locally.

## Project Structure

```text
Financial-News-Sentiment-Stock-Market-Correlation-Analysis-main/
|-- README.md
|-- requirements.txt
|-- .env
|-- data/
|   |-- news_cache.csv
|   `-- stock_cache.csv
|-- notebooks/
|   `-- Correlation_analysis.ipynb
|-- scripts/
|   `-- task-3main.py
`-- src/
    |-- data_fetcher.py
    |-- data_cleaner.py
    |-- data_merger.py
    |-- sentiment_analysis.py
    |-- stock_returns.py
    |-- correlation_analysis.py
    `-- visualizationt.py
```

## File Responsibilities

- `scripts/task-3main.py`: Main runnable pipeline.
- `notebooks/Correlation_analysis.ipynb`: Interactive analysis notebook with intermediate outputs and per-company charts.
- `src/data_fetcher.py`: Downloads news and stock data, handles caching, and generates fallback sample news.
- `src/data_cleaner.py`: Parses dates, removes invalid rows, and formats date columns.
- `src/data_merger.py`: Merges stock and news data by date.
- `src/sentiment_analysis.py`: Converts headlines into TextBlob polarity scores and aggregates sentiment by day.
- `src/stock_returns.py`: Computes daily percentage returns from closing prices.
- `src/correlation_analysis.py`: Calculates Pearson correlation.
- `src/visualizationt.py`: Builds the overall charts.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd Financial-News-Sentiment-Stock-Market-Correlation-Analysis-main
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Add your Alpha Vantage API key:

```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Requirements

Main packages used by this project:

- pandas
- matplotlib
- seaborn
- nltk
- textblob
- requests
- python-dotenv
- yfinance

## How To Run

### Run the script pipeline

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\task-3main.py
```

This will:

- fetch or load cached data
- compute sentiment and returns
- print the Pearson correlation coefficient
- show the overall visualizations

### Run the notebook

Open the notebook below in VS Code or Jupyter and run the cells in order:

- `notebooks/Correlation_analysis.ipynb`

The notebook includes:

- setup and cache loading
- data inspection
- merged dataset construction
- sentiment scoring
- overall correlation
- per-company correlation plots

## Output

The project produces:

- a merged news and stock analysis dataset
- overall daily return and sentiment plots
- an overall sentiment vs return scatter plot
- per-company scatter plots for all tracked companies
- Pearson correlation values overall and by company

## Example Analysis Question

Does more positive news sentiment on a given day align with stronger stock returns for the same day?

This repository is built to answer that question programmatically and visually.

## Notes And Limitations

- Free Alpha Vantage access is rate-limited.
- If fallback sample news is used, the resulting correlation values are only suitable for demo or development purposes.
- The current sentiment model uses headline text only, not full article content.
- Pearson correlation captures linear association only and does not imply causation.

## Suggested Report Use

This project is suitable for an academic or portfolio report that includes:

- project background and objective
- dataset description
- methodology
- overall results
- per-company graph analysis
- limitations and recommendations

## Contributing

Contributions are welcome through issues or pull requests.

## License

No license file is currently included in the repository. Add one if you intend to share or reuse this project publicly.
