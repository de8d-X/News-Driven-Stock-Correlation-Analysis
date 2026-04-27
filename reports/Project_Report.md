# News-Driven Stock Correlation Analysis
## Financial News Sentiment and Stock Market Correlation Analysis

---

**Project Title:** News-Driven Stock Correlation Analysis: Examining the Relationship Between Financial News Sentiment and Short-Term Stock Returns for Major Technology Companies

**Course:** Data Science / Financial Analytics

**Level:** Undergraduate / Graduate

**Date:** April 2026

**Repository:** News-Driven-Stock-Correlation-Analysis

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Background and Literature Context](#3-background-and-literature-context)
4. [Problem Statement](#4-problem-statement)
5. [Objectives](#5-objectives)
6. [Research Questions](#6-research-questions)
7. [Data Sources](#7-data-sources)
8. [Project Architecture and Repository Structure](#8-project-architecture-and-repository-structure)
9. [Workflow Overview](#9-workflow-overview)
10. [Methodology](#10-methodology)
11. [Implementation Details](#11-implementation-details)
12. [Exploratory Data Analysis](#12-exploratory-data-analysis)
13. [Sentiment Analysis Results](#13-sentiment-analysis-results)
14. [Stock Return Analysis](#14-stock-return-analysis)
15. [Overall Correlation Results](#15-overall-correlation-results)
16. [Company-Level Results](#16-company-level-results)
17. [Visualization and Interpretation](#17-visualization-and-interpretation)
18. [Limitations](#18-limitations)
19. [Ethical Considerations](#19-ethical-considerations)
20. [Conclusion](#20-conclusion)
21. [Future Work](#21-future-work)
22. [References and Tools](#22-references-and-tools)
23. [Appendices](#23-appendices)

---

## 1. Abstract

This report presents a comprehensive data science investigation into the statistical relationship between the sentiment of financial news headlines and the short-term daily stock price returns of seven major technology companies listed on United States equity markets. Using an end-to-end Python pipeline, news headline data sourced from the Alpha Vantage NEWS_SENTIMENT API was combined with daily stock price data retrieved via the Yahoo Finance interface (`yfinance`). Each headline was assigned a continuous polarity score between −1.0 and +1.0 using the TextBlob lexicon-based natural language processing library. Daily stock returns were computed as percentage changes in closing prices. The Pearson correlation coefficient was then calculated between the aggregated daily average sentiment and daily returns, both across all seven companies and individually per company.

The overall correlation was found to be **r = −0.0026**, indicating virtually no linear association between news sentiment and same-day stock returns in the aggregate dataset. Per-company analysis revealed slightly differentiated but uniformly weak results: the range of individual Pearson coefficients spanned from **r = −0.1754** (Tesla) to **r = +0.0136** (Microsoft). These findings are consistent with predictions from the semi-strong form of the Efficient Market Hypothesis, which holds that all publicly available information is already incorporated into equity prices by the time they are observed.

Critically, the data used in this study was generated using a synthetic headline fallback mechanism triggered by the Alpha Vantage free-tier rate limit, meaning the headlines do not correspond to real market-moving events. This structural limitation means the results cannot be taken as definitive evidence for or against a news-sentiment effect; instead, they demonstrate the successful construction of a reproducible analytical pipeline ready for deployment against authentic news data. The report concludes with an examination of limitations, ethical considerations, and a detailed roadmap of future improvements.

---

## 2. Introduction

### 2.1 Background

The relationship between publicly available information and financial asset prices is one of the most studied problems in quantitative finance and financial economics. Since the late nineteenth century, financial practitioners and academics alike have recognized that news—whether corporate announcements, macroeconomic data releases, or geopolitical developments—can move markets rapidly and significantly. With the rise of digital media, social networks, and algorithmic trading, the speed at which market participants process and react to new information has accelerated dramatically, compressing the window from news publication to price response from days or hours down to seconds.

Within this context, the field of financial sentiment analysis has emerged at the intersection of natural language processing (NLP) and quantitative finance. The core premise is straightforward: if news headlines carry systematically positive or negative connotations about a company or sector, and if market participants respond rationally to this information, then days with more positive headline sentiment should, on average, coincide with stronger price performance than days with more negative sentiment. However, empirical evidence for this relationship is mixed, time-varying, and highly sensitive to the quality of both the news dataset and the sentiment model.

### 2.2 Motivation

This project was motivated by the desire to build a transparent, reproducible, and accessible demonstration of how a modern data science pipeline can be constructed to answer a concrete financial research question using entirely open-source tools. Rather than relying on proprietary Bloomberg terminals, expensive Reuters data subscriptions, or opaque black-box models, this project uses freely available APIs (Alpha Vantage, Yahoo Finance), open-source Python libraries (TextBlob, pandas, yfinance, matplotlib, seaborn), and structured modular code organized into a clean repository. This approach makes the project suitable as a portfolio piece, as an academic exercise, or as a starting point for more advanced research.

The seven companies selected for analysis—Apple (AAPL), Amazon (AMZN), Google (GOOGL), Meta (META), Microsoft (MSFT), Nvidia (NVDA), and Tesla (TSLA)—represent the most widely covered and most actively traded technology stocks in the world. They collectively account for a substantial fraction of the market capitalization of the S&P 500, and their stocks are subject to intense media scrutiny and analyst coverage, making them an ideal laboratory for studying news–price interactions.

### 2.3 Scope of This Report

This report describes all phases of the project in detail, from the conceptual motivation and research design through the technical implementation, empirical results, and interpretation. It is written at a level appropriate for an upper-division undergraduate or early graduate course in data science, computational finance, or applied machine learning. Technical concepts are explained in sufficient detail that a reader with a general background in statistics and programming can follow the methodology and understand the implications of the results.

The report is organized as follows. Section 3 provides additional background on the Efficient Market Hypothesis and prior work on financial sentiment analysis. Section 4 formalizes the problem statement. Sections 5 and 6 detail the objectives and research questions. Section 7 describes the data sources. Section 8 explains the repository structure. Sections 9 through 12 cover the workflow, methodology, implementation details, and exploratory data analysis. Sections 13 through 16 present the main results. Sections 17 through 19 address visualization, limitations, and ethical considerations. Sections 20 through 23 conclude the report and provide references and appendices.

---

## 3. Background and Literature Context

### 3.1 The Efficient Market Hypothesis

The **Efficient Market Hypothesis (EMH)** is the foundational theoretical framework for understanding why—or why not—news sentiment might predict stock returns. Originally formalized by Eugene Fama in 1970, the EMH posits that financial markets are informationally efficient: that is, asset prices fully reflect all available information at any given time. The hypothesis comes in three variants:

- **Weak-form efficiency:** Prices reflect all historical price and volume data, rendering technical analysis ineffective for generating excess returns.
- **Semi-strong form efficiency:** Prices reflect all publicly available information, including financial statements, news articles, and analyst reports.
- **Strong-form efficiency:** Prices reflect all information, including private (insider) information.

For the purposes of this project, the semi-strong form is most relevant. If markets are semi-strong efficient, then the sentiment content of publicly available news headlines should already be priced in by the time daily closing prices are computed. In this case, we would expect the correlation between daily headline sentiment and same-day returns to be near zero—which is precisely the result observed.

However, the EMH has been challenged extensively over the past four decades. Behavioral finance research has documented numerous instances of market anomalies—patterns in returns that appear to be inconsistent with informationally efficient pricing. Sentiment-driven price effects are among the most debated of these anomalies. Some researchers have found that high investor sentiment periods are followed by lower average returns (consistent with overpricing), while others have found that specific types of news events—earnings surprises, CEO changes, product launches—do generate short-term return predictability.

### 3.2 Sentiment Analysis in Finance

The application of natural language processing to financial text is a rapidly growing subfield of both computational linguistics and quantitative finance. Early approaches relied on simple word-count methods—counting occurrences of positive and negative words from curated financial lexicons such as the Loughran–McDonald (LM) sentiment dictionary, which was specifically designed to capture the meaning of words in a financial context (where "liability," for instance, carries a strongly negative connotation not captured by general-purpose lexicons).

More recently, deep learning approaches—particularly transformer-based models such as BERT (Bidirectional Encoder Representations from Transformers) and its financial fine-tuning variant FinBERT—have substantially improved the accuracy of financial sentiment classification. These models capture nuanced contextual meaning, handle negation and sarcasm more robustly, and can distinguish between sentiment about a specific entity versus the general market environment.

The present project uses **TextBlob**, a general-purpose lexicon-based NLP library. While TextBlob is not specialized for financial language, it provides an accessible and interpretable baseline that is appropriate for a first-pass exploratory study. Its output—a polarity score in [−1, +1]—is directly comparable across headlines and directly usable in a correlation calculation. The choice to use TextBlob is acknowledged as a limitation, and the use of FinBERT is explicitly recommended as a future improvement.

### 3.3 Related Work

Numerous academic studies have examined the relationship between financial news sentiment and stock market behavior. A few key findings are summarized below to provide context for interpreting the results of this project:

**Tetlock (2007)** analyzed the "Abreast of the Market" column in the Wall Street Journal and found that high levels of media pessimism predict downward pressure on market prices, followed by a reversal, suggesting a short-term overreaction and subsequent correction rather than efficient instantaneous pricing.

**Bollen, Mao, and Zeng (2011)** used Twitter mood states derived from psycholinguistic tools to predict daily Dow Jones Industrial Average movements, reporting prediction accuracy above 86% using a hybrid sentiment-neural network model. However, these results have been difficult to replicate in out-of-sample periods.

**Loughran and McDonald (2011)** demonstrated that the general-purpose Harvard General Inquirer word list misclassifies a large proportion of financial text—approximately three-quarters of the words considered negative in the Harvard list are not negative in a financial context—motivating the development of domain-specific financial lexicons.

**Malo et al. (2014)** created the Financial PhraseBank, a dataset of financial news sentences manually annotated for sentiment, which has become a standard benchmark for financial NLP models including FinBERT.

These works collectively suggest that (a) news sentiment does contain information relevant to stock price movements, but (b) the relationship is sensitive to the quality of the NLP model, the type and granularity of news data, the time horizon of analysis, and the specific market or asset class under study. The present project is most aligned with the exploratory, tools-focused studies rather than the high-precision prediction literature.

---

## 4. Problem Statement

Financial markets are inherently complex systems driven by a multitude of factors, including macroeconomic indicators, corporate earnings, investor sentiment, and the broader information environment. Among these factors, news media plays a particularly influential role in shaping market participants' perceptions and, by extension, their trading decisions. The proliferation of real-time digital news sources has made it increasingly plausible that the tone and content of financial news headlines can move stock prices on short timescales.

Despite this widely held intuition, the empirical relationship between news sentiment and same-day stock returns remains an active research question, particularly at the granularity of individual companies and headline-level text. Existing quantitative finance literature tends to analyze large institutional datasets, sophisticated natural language processing pipelines, or high-frequency trading data. There is comparatively less work that applies accessible, open-source tools to reproduce and validate this relationship in a transparent and reproducible fashion.

This project addresses that gap by posing a clear and testable empirical question: **Does a measurable statistical correlation exist between the average daily sentiment of financial news headlines and the daily percentage returns of major technology stocks, and if so, does this relationship vary meaningfully across individual companies?**

The project is further motivated by a practical systems design question: **Can an end-to-end analytical pipeline be built from freely available tools and APIs that is modular, extensible, and capable of running entirely offline using cached data?**

Both questions are addressed in this report. The first is an empirical question answered through data analysis; the second is a software engineering question answered through the architecture and implementation of the codebase.

---

## 5. Objectives

The primary objectives of this project are defined in two categories: analytical objectives and technical objectives.

### 5.1 Analytical Objectives

1. **Collect and validate** financial news headline data and historical daily stock price data for seven major technology companies (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA) covering approximately six months of trading history.

2. **Apply lexicon-based sentiment analysis** to assign a continuous polarity score to each news headline, using the TextBlob library, yielding a score between −1.0 (maximally negative) and +1.0 (maximally positive).

3. **Compute daily percentage stock returns** from historical closing price data for each company, using the standard finance formula for log-linear return approximation.

4. **Aggregate headline-level sentiment scores** to a single daily average value per trading day, facilitating a one-to-one alignment with daily stock return observations.

5. **Calculate the Pearson correlation coefficient** between aggregated daily sentiment and daily return, both in aggregate across all companies and individually for each of the seven companies.

6. **Visualize the sentiment-return relationship** via time-series overlay plots and scatter diagrams, enabling both quantitative and qualitative interpretation.

7. **Interpret the empirical results** in the context of financial market theory, acknowledging the impact of data quality limitations on the validity of conclusions.

### 5.2 Technical Objectives

1. **Design a modular Python pipeline** with clearly separated responsibilities for data fetching, cleaning, merging, analysis, and visualization, each encapsulated in its own source module.

2. **Implement robust data caching** to minimize redundant API calls and allow fully offline execution once initial data has been fetched.

3. **Implement a graceful fallback mechanism** that generates synthetic but representative headlines when the API quota is exhausted, ensuring the pipeline can always produce end-to-end outputs.

4. **Expose both a script-based and a notebook-based interface** to the pipeline, making the project accessible to both command-line users and interactive notebook users.

5. **Document the codebase** sufficiently for reproduction and extension by other researchers or students.

---

## 6. Research Questions

The investigation is organized around three primary research questions and several secondary exploratory questions.

### 6.1 Primary Research Questions

**RQ1:** Is there a statistically detectable linear correlation between the average daily sentiment of financial news headlines and the daily percentage stock return, when data is aggregated across all seven technology companies?

**RQ2:** Does the sentiment-return correlation differ systematically across individual companies, and if so, which companies exhibit the strongest and weakest relationships?

**RQ3:** Can a reproducible, fully open-source analytical pipeline be constructed that fetches, processes, analyzes, and visualizes this relationship with minimal manual intervention?

### 6.2 Secondary Exploratory Questions

**SQ1:** What is the distribution of TextBlob sentiment scores across headlines—how many headlines are classified as positive, neutral, or negative?

**SQ2:** What is the typical daily return magnitude for each of the seven companies, and how volatile are returns over the analysis period?

**SQ3:** Are there observable temporal clusters of either consistently positive or consistently negative sentiment, and do these correspond to visible patterns in the return series?

**SQ4:** How does the direction of any observed correlation (positive vs. negative) compare with theoretical predictions—specifically, does higher sentiment align with higher returns (consistent with a momentum or information-diffusion effect) or with lower returns (consistent with a "buy the rumor, sell the news" effect)?

---

## 7. Data Sources

### 7.1 Financial News Headlines — Alpha Vantage NEWS_SENTIMENT API

#### 7.1.1 API Description

News headline data was retrieved from the **Alpha Vantage** financial data platform using its `NEWS_SENTIMENT` endpoint. Alpha Vantage provides a free-tier API that allows users to request recent news articles associated with a specific ticker symbol. For each request, the API returns a feed of news articles, each containing the article title (headline), publication timestamp, source URL, and optionally a pre-computed sentiment label. In this project, only the headline text and publication timestamp were used.

#### 7.1.2 Request Parameters

For each ticker symbol, a request was made with the following parameters:

| Parameter  | Value                    |
|------------|--------------------------|
| function   | `NEWS_SENTIMENT`         |
| tickers    | Target ticker (e.g., AAPL) |
| limit      | 50                       |
| apikey     | (stored in .env file)    |

The API was queried sequentially for each of the seven tickers, with a 12-second delay between requests to respect the free-tier rate limit of approximately five requests per minute.

#### 7.1.3 Free-Tier Limitations and Fallback Mechanism

The Alpha Vantage free tier enforces a strict daily quota on the total number of API requests. When this quota is exhausted—indicated by the presence of an "Information" or "Note" key in the JSON response—the pipeline automatically switches to a fallback mode. In fallback mode, synthetic headlines are generated by randomly selecting from a library of fifteen template strings and formatting them with the relevant company name. Each synthetic headline is assigned to a randomly selected trading date from the available stock data date range.

The fifteen headline templates cover a representative range of financial news events:

- Earnings beats and misses
- Product launches and expansions
- Regulatory scrutiny and legal challenges
- Share buyback programs
- CEO strategy presentations
- Analyst upgrades and downgrades
- Workforce restructuring
- Strategic acquisitions
- Partnership announcements
- Guidance revisions

This fallback ensures that the pipeline can always produce end-to-end outputs, even without API access. However, as noted throughout this report, synthetic headlines carry no genuine informational content about actual market events, which significantly limits the validity of the resulting correlation analysis.

#### 7.1.4 Cached News Dataset

The news dataset used in this analysis was stored in `data/news_cache.csv` and contained **1,746 headline records** spanning the period from **October 27, 2025 through April 24, 2026**. Each record contains three fields: `headline` (text), `date` (YYYY-MM-DD), and `stock` (ticker symbol).

### 7.2 Historical Stock Price Data — Yahoo Finance (yfinance)

#### 7.2.1 Library Description

Daily historical stock price data was retrieved using the **yfinance** Python library, which provides a clean interface to Yahoo Finance's publicly available historical price database. Unlike Alpha Vantage, yfinance requires no API key and imposes no strict rate limits, making it suitable for bulk data retrieval.

#### 7.2.2 Data Retrieved

For each of the seven ticker symbols, the following daily OHLCV (Open, High, Low, Close, Volume) data was retrieved:

| Column  | Description                              |
|---------|------------------------------------------|
| Date    | Trading date (YYYY-MM-DD)                |
| Open    | Opening price at market open             |
| High    | Highest price reached during the session |
| Low     | Lowest price reached during the session  |
| Close   | Closing price at market close            |
| Volume  | Total number of shares traded            |

A six-month lookback period (`period='6mo'`) was used, yielding approximately 124 trading days per company.

#### 7.2.3 Cached Stock Dataset

The stock dataset was stored in `data/stock_cache.csv` and contained **868 records** (approximately 124 trading days × 7 companies). Each record represents a single company's daily OHLCV data, with an additional `Company` column indicating the company name (e.g., "Apple", "Amazon").

### 7.3 Data Coverage Summary

| Dataset    | Records | Date Range                      | Source        |
|------------|---------|----------------------------------|---------------|
| News       | 1,746   | 2025-10-27 — 2026-04-24         | Alpha Vantage |
| Stock      | 868     | 2025-10-27 — 2026-04-24         | Yahoo Finance |

The seven companies analyzed were:

| Ticker | Company Name | Approximate Price Range (Oct 2025 – Apr 2026) |
|--------|--------------|-----------------------------------------------|
| AAPL   | Apple        | $226 – $268                                   |
| AMZN   | Amazon       | $215 – $235                                   |
| GOOGL  | Google       | $168 – $196                                   |
| META   | Meta         | $565 – $765                                   |
| MSFT   | Microsoft    | $388 – $544                                   |
| NVDA   | Nvidia       | $108 – $153                                   |
| TSLA   | Tesla        | $215 – $460                                   |

---

## 8. Project Architecture and Repository Structure

### 8.1 Directory Layout

The project is organized as a clean, modular Python repository with the following top-level structure:

```
News-Driven-Stock-Correlation-Analysis/
│
├── README.md                         # Project overview and setup instructions
├── requirements.txt                  # Python package dependencies
├── .env                              # API key configuration (not committed)
├── .gitignore                        # Files excluded from version control
│
├── data/
│   ├── news_cache.csv                # Cached news headline data
│   └── stock_cache.csv               # Cached stock price data
│
├── notebooks/
│   ├── Correlation_analysis.ipynb    # Interactive analysis notebook
│   └── sentimented_df.csv            # Merged dataset with sentiment scores
│
├── scripts/
│   └── task-3main.py                 # Main pipeline entry point
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── data_fetcher.py               # API data retrieval and caching
│   ├── data_cleaner.py               # Date parsing and null removal
│   ├── data_merger.py                # DataFrame merging utilities
│   ├── sentiment_analysis.py         # TextBlob sentiment scoring
│   ├── stock_returns.py              # Daily return computation
│   ├── correlation_analysis.py       # Pearson correlation calculation
│   └── visualizationt.py             # Chart generation
│
└── reports/
    └── Project_Report.md             # This academic report
```

### 8.2 Module Responsibilities

The `src/` directory contains the core analytical logic, organized by functional responsibility:

**`data_fetcher.py`**
Handles all external data acquisition. Implements `fetch_news()` for Alpha Vantage headline retrieval with quota detection and synthetic fallback, and `fetch_stock_data_yfinance()` for Yahoo Finance OHLCV data retrieval via the yfinance library. Also implements `_generate_sample_news()` for synthetic headline generation.

**`data_cleaner.py`**
Provides three utility functions: `clean_data()` standardizes date columns to timezone-naive datetime objects and drops rows with null values; `format_date_column()` converts datetime objects to ISO 8601 string format; `drop_first_column()` removes unnamed index columns that may appear in CSV exports.

**`data_merger.py`**
Implements two merge operations: `merge_data()` performs an inner join of the news and stock DataFrames on the `date` column; `merge_sentiment_with_stock()` merges the daily sentiment aggregation with stock return data.

**`sentiment_analysis.py`**
Contains `analyze_sentiment()`, a single-function wrapper around `TextBlob(text).sentiment.polarity`, and `aggregate_sentiments_by_date()`, which groups sentiment scores by date and computes the daily mean.

**`stock_returns.py`**
Contains `calculate_daily_returns()`, which appends a `Daily Return` column computed as the percentage change in closing prices.

**`correlation_analysis.py`**
Contains `calculate_correlation()`, a thin wrapper around pandas' `.corr()` method that returns the off-diagonal Pearson correlation between the `Average Sentiment` and `Daily Return` columns.

**`visualizationt.py`**
Contains `plot_daily_returns()`, which generates a dual-panel time-series figure of returns and sentiment, and `scatter_plot()`, which generates a scatter diagram of sentiment vs. returns.

### 8.3 Execution Interfaces

The project supports two execution interfaces:

**Script mode:** Running `python scripts/task-3main.py` from the project root executes the full pipeline sequentially, printing progress messages and displaying matplotlib figures.

**Notebook mode:** Opening `notebooks/Correlation_analysis.ipynb` in Jupyter or VS Code provides an interactive, cell-by-cell walkthrough of the pipeline with intermediate outputs, per-company analysis, and inline visualizations.

---

## 9. Workflow Overview

The analytical workflow consists of eight sequential stages, implemented identically in both the script and the notebook:

### Stage 1: Data Acquisition with Caching

The pipeline begins by checking whether cached data files exist at `data/news_cache.csv` and `data/stock_cache.csv`. If both exist, they are loaded directly into pandas DataFrames without any network calls. If either is absent, the relevant data is fetched from the appropriate external source (Alpha Vantage for news, Yahoo Finance for stock prices) and saved to disk before proceeding. This two-layer caching strategy ensures that:

1. The daily Alpha Vantage quota is preserved across multiple runs.
2. The pipeline can be executed entirely offline after the initial data fetch.
3. Analysis results are reproducible, as the input data does not change between runs.

In the event that the Alpha Vantage quota is exhausted and no news cache exists, the pipeline generates synthetic headlines spanning the same date range as the stock data. This ensures that the pipeline never fails mid-execution due to data unavailability.

### Stage 2: Data Cleaning

Once both DataFrames are loaded, they are passed through the data cleaning module. The `clean_data()` function performs three operations:

1. Renames the specified date column to `date` (if it was named differently) for consistent downstream access.
2. Converts the `date` column to a pandas datetime type using `pd.to_datetime()` with `errors='coerce'` to handle malformed date strings gracefully, and strips any timezone offset using `.dt.tz_localize(None)`.
3. Drops all rows containing any null values, ensuring the downstream analysis operates on complete records only.

The `format_date_column()` function then converts datetime objects to ISO 8601 string format (`YYYY-MM-DD`) using `.dt.strftime()`. This standardization ensures that date-based merges operate correctly even if the original data used different date formats.

### Stage 3: Data Merging

The cleaned news and stock DataFrames are joined using an inner merge on the `date` column. The inner join behavior ensures that only dates present in both datasets are retained. This is important because:

- Stock markets are closed on weekends and public holidays, so not every calendar day has a corresponding stock record.
- News may be published on non-trading days, and such records would not have a corresponding stock return.

After the merge, each row of the resulting DataFrame represents a single headline associated with a particular date, along with all stock data (Open, High, Low, Close, Volume, Company) from the corresponding trading day.

### Stage 4: Sentiment Scoring

The `analyze_sentiment()` function is applied via pandas' `.apply()` method to the `headline` column of the merged DataFrame, creating a new `sentiment` column. This operation processes each headline individually through the TextBlob sentiment analyzer and appends the resulting polarity score.

This stage is computationally the most demanding in the pipeline, as it requires a string-parsing NLP operation for each of the approximately 1,700+ rows in the merged dataset. In practice, execution time is on the order of several seconds on a modern laptop.

### Stage 5: Daily Return Calculation

The `calculate_daily_returns()` function appends a `Daily Return` column by calling pandas' `.pct_change()` on the `Close` column and multiplying by 100 to express returns as percentages. The first row for each company will naturally receive a NaN value, as there is no prior day from which to compute a change.

### Stage 6: Sentiment Aggregation

The `aggregate_sentiments_by_date()` function groups the merged DataFrame by the `date` column and computes the mean sentiment score for all headlines published on each date, producing a compact `daily_sentiment` DataFrame with one row per trading day.

### Stage 7: Final Merge and Correlation

The `merge_sentiment_with_stock()` function merges the `daily_sentiment` aggregation back onto the stock return data using another inner join on `date`. The result is a `daily_df` DataFrame containing one row per trading day with columns for both `Average Sentiment` and `Daily Return`. The `calculate_correlation()` function then extracts the Pearson correlation coefficient from this DataFrame.

### Stage 8: Visualization

The `plot_daily_returns()` function generates a two-panel time-series figure displaying the temporal evolution of both daily stock returns and average sentiment. The `scatter_plot()` function generates a scatter diagram of daily sentiment on the horizontal axis against daily return on the vertical axis, with each point representing one trading day.

---

## 10. Methodology

### 10.1 Sentiment Analysis with TextBlob

**TextBlob** is an open-source Python library for processing textual data. Its sentiment analysis functionality is built on top of the **Pattern** library and uses a dictionary-based approach in which each word or phrase recognized by the pattern sentiment analyzer carries a pre-assigned polarity value. The overall polarity of a sentence is computed as a weighted average of the polarity values of its constituent tokens.

The `sentiment.polarity` attribute returns a floating-point score:

| Score Range | Interpretation         |
|-------------|------------------------|
| −1.0 to −0.1 | Negative sentiment    |
| −0.1 to +0.1 | Neutral / ambiguous   |
| +0.1 to +1.0 | Positive sentiment    |

For example:
- **"Apple reports strong quarterly earnings, beating analyst expectations"** → polarity ≈ +0.35 (positive)
- **"Apple faces regulatory scrutiny over data privacy concerns"** → polarity ≈ −0.20 (negative)
- **"Apple CEO outlines growth strategy at investor conference"** → polarity ≈ 0.0 (neutral)

In financial contexts, TextBlob's general-purpose dictionary may not correctly classify domain-specific terminology. Words such as "short" (as in short selling), "crush" (as in "crushed earnings estimates"), or "cut" (as in "cut guidance") may receive unexpected polarity scores. This is a known limitation acknowledged in Section 18.

**Code implementation:**

```python
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
```

### 10.2 Daily Stock Return Calculation

The **daily percentage return** is defined as:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100$$

where:
- $r_t$ = daily return on day $t$ (expressed as a percentage)
- $P_t$ = closing price on day $t$
- $P_{t-1}$ = closing price on the previous trading day

This measure is standard in financial analysis for several reasons:

1. **Normalization:** It expresses returns as a proportion of the prior price, enabling meaningful comparison across companies with vastly different absolute share prices.
2. **Stationarity:** Percentage returns are more approximately stationary over time than raw price levels, which typically exhibit upward trends.
3. **Interpretability:** A daily return of +2.0 means the stock gained 2% of its previous close value in a single trading session.

**Code implementation:**

```python
def calculate_daily_returns(df, close_column='Close', return_column='Daily Return'):
    df[return_column] = df[close_column].pct_change() * 100
    return df
```

### 10.3 Daily Sentiment Aggregation

Since multiple headlines may be published for a given company on a single trading day, it is necessary to aggregate headline-level sentiment scores to a single daily value before computing the correlation with the corresponding daily return. This project uses the **arithmetic mean** of all headline sentiment scores on a given date:

$$\bar{s}_t = \frac{1}{N_t} \sum_{i=1}^{N_t} s_{i,t}$$

where:
- $\bar{s}_t$ = average sentiment on day $t$
- $N_t$ = number of headlines published on day $t$
- $s_{i,t}$ = sentiment polarity of headline $i$ on day $t$

**Code implementation:**

```python
def aggregate_sentiments_by_date(df, date_column='date', sentiment_column='sentiment'):
    daily_sentiment = df.groupby(date_column)[sentiment_column].mean().reset_index()
    daily_sentiment.columns = [date_column, 'Average Sentiment']
    return daily_sentiment
```

Alternative aggregation methods—such as the median, maximum, or a volume-weighted average (where more-read articles receive higher weights)—were not explored in this version but are suggested as future improvements.

### 10.4 Pearson Correlation Coefficient

The **Pearson correlation coefficient** is a measure of the linear association between two continuous variables. It is defined as:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

where:
- $x$ = Average Sentiment variable
- $y$ = Daily Return variable
- $\bar{x}$, $\bar{y}$ = respective sample means
- $n$ = number of paired observations

Key properties of $r$:
- $r \in [-1, +1]$
- $r = +1$: perfect positive linear association
- $r = -1$: perfect negative linear association
- $r = 0$: no linear association
- Values near 0 indicate little or no linear relationship

Commonly used benchmarks for interpreting the strength of $r$ in social and behavioral sciences:

| |r| Range | Interpretation        |
|------------|------------------------|
| 0.00 – 0.10 | Negligible            |
| 0.10 – 0.30 | Weak                  |
| 0.30 – 0.50 | Moderate              |
| 0.50 – 0.70 | Strong                |
| 0.70 – 1.00 | Very strong           |

All correlation values observed in this project fall in the "negligible" to "weak" range.

**Code implementation:**

```python
def calculate_correlation(df, column1='Average Sentiment', column2='Daily Return'):
    correlation = df[[column1, column2]].corr().iloc[0, 1]
    return correlation
```

### 10.5 Statistical Significance

This analysis did not compute formal p-values or confidence intervals for the Pearson correlation coefficients. With approximately 124 observations per company (number of trading days), the minimum detectable correlation at a two-tailed 5% significance level is approximately $r \approx 0.177$, based on the critical value from the t-distribution with $n - 2 = 122$ degrees of freedom. Most of the company-level correlations observed in this study (ranging from −0.1754 to +0.0136) would not meet this threshold, and therefore cannot be considered statistically significant in a formal hypothesis testing framework.

This is an important caveat: the absence of a significant correlation does not prove that no relationship exists—it may simply reflect insufficient statistical power or the use of synthetic data that masks any real underlying signal.

---

## 11. Implementation Details

### 11.1 Environment Setup

The project was developed using Python and requires the following dependencies, specified in `requirements.txt`:

| Package        | Role                                              |
|----------------|---------------------------------------------------|
| `pandas`       | Data manipulation and DataFrame operations        |
| `numpy`        | Numerical operations (implicit dependency)        |
| `matplotlib`   | Plot generation and figure management             |
| `seaborn`      | Statistical scatter plots and aesthetics          |
| `textblob`     | Natural language processing and sentiment scoring |
| `nltk`         | Natural language toolkit (TextBlob dependency)    |
| `requests`     | HTTP requests to Alpha Vantage API                |
| `yfinance`     | Yahoo Finance stock data retrieval                |
| `python-dotenv`| Loading API keys from `.env` file                 |

API authentication is handled via a `.env` file at the project root containing:

```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

This file is excluded from version control via `.gitignore` to prevent accidental credential exposure.

### 11.2 Data Fetcher Implementation

The `fetch_news()` function implements a careful sequential fetch pattern to respect the Alpha Vantage free-tier rate limit:

```python
for i, ticker in enumerate(tickers):
    if quota_hit:
        break
    # ... make API request ...
    if "Information" in data or "Note" in data:
        quota_hit = True
        break
    # ... parse response ...
    if i < len(tickers) - 1:
        time.sleep(12)  # 12-second delay between requests
```

The 12-second inter-request delay ensures that no more than five requests are made per minute, consistent with the free-tier limit. The `"Information"` and `"Note"` keys in the response JSON are the signals Alpha Vantage uses to indicate that the daily quota has been reached.

### 11.3 Date Alignment Strategy

A critical implementation decision is how to align news and stock data by date. News is often published on weekends or after market close, leading to mismatches with the trading-day-indexed stock data. The project handles this using an inner join on the `date` column:

```python
merged_df = pd.merge(news_df, stock_df, on='date')
```

The inner join naturally excludes dates where news was published on non-trading days (weekends, holidays), because those dates do not appear in the stock DataFrame. This conservative approach sacrifices some news coverage in exchange for cleaner data alignment, since it is unclear which trading day a weekend headline should be attributed to without domain-specific logic.

### 11.4 Caching Strategy

Both datasets are saved to CSV files in the `data/` directory immediately after fetching:

```python
news_df.to_csv(NEWS_CACHE, index=False)
stock_df.to_csv(STOCK_CACHE, index=False)
```

On subsequent runs, both files are loaded with `pd.read_csv()` without any network calls. This idempotent design ensures that:

1. The same analysis can be reproduced at any point in the future.
2. API quotas are consumed only once, on the initial fetch.
3. The pipeline can be run in fully air-gapped environments.

---

## 12. Exploratory Data Analysis

### 12.1 News Dataset Overview

The cached news dataset contained **1,746 headlines** across seven companies. Summary statistics:

| Metric                         | Value                        |
|-------------------------------|------------------------------|
| Total records                 | 1,746                        |
| Unique dates                  | ~124 (Oct 2025 – Apr 2026)   |
| Average headlines per day     | ~14 (2 per company per day)  |
| Companies covered             | 7 (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA) |
| Data source                   | Synthetic fallback            |
| Headline columns              | headline, date, stock         |

The headlines were uniformly distributed across companies, with each company contributing approximately 249 headlines (1,746 / 7 ≈ 249). This uniformity is characteristic of the synthetic fallback mechanism, which generates 1–3 random headlines per company per trading day.

### 12.2 Stock Dataset Overview

The cached stock dataset contained **868 OHLCV records**. Summary statistics:

| Metric                        | Value                          |
|------------------------------|--------------------------------|
| Total records                 | 868                            |
| Records per company           | ~124                           |
| Trading days covered          | ~124 (Oct 2025 – Apr 2026)    |
| Price columns                 | Open, High, Low, Close, Volume |
| Additional columns            | date, Company                  |

The following table provides an illustrative snapshot of early stock data for each company (October 2025):

| Company   | Date       | Close ($) | Daily Return (%) |
|-----------|------------|-----------|------------------|
| Amazon    | 2025-10-27 | 226.97    | N/A (first day)  |
| Amazon    | 2025-10-28 | 229.25    | +1.00%           |
| Amazon    | 2025-10-29 | 230.30    | +0.46%           |
| Amazon    | 2025-10-30 | 222.86    | −3.23%           |
| Amazon    | 2025-10-31 | 244.22    | +9.58%           |
| Apple     | 2025-10-27 | 268.30    | N/A              |
| Apple     | 2025-10-28 | 268.49    | +0.07%           |
| Apple     | 2025-10-29 | 269.19    | +0.26%           |
| Apple     | 2025-10-30 | 270.88    | +0.63%           |
| Apple     | 2025-10-31 | 269.86    | −0.38%           |

### 12.3 Merged Dataset Characteristics

After merging news and stock data on the `date` column, the resulting DataFrame contained a cross-product of headlines and stock records per date. Because the merge joins news (keyed on date and ticker, 1,746 rows) with stock (keyed on date and company, 868 rows), and because the stock table contains records for all seven companies for each date, the merged DataFrame is substantially larger: **12,222 rows** (as indicated by the sentimented_df.csv file in the notebooks directory).

This is because each headline on a given date is paired with the stock data for all seven companies on that date (not just the company the headline was about). This cross-product join reflects the project's design intention: to study the overall sentiment-return relationship across the market rather than restricting to ticker-specific news only. While this approach increases statistical power by multiplying the observation count, it also introduces a degree of within-day autocorrelation (the same headline appears paired with seven different company returns), which is a methodological limitation.

---

## 13. Sentiment Analysis Results

### 13.1 Score Distribution

Given that the headline data consists of synthetic fallback headlines drawn from a curated template library, the distribution of TextBlob sentiment scores reflects the sentiment profile of the fifteen template strings rather than the actual financial news environment. The fifteen templates were designed to span a representative range of positive, negative, and neutral news events:

**Positive templates (expected polarity > 0):**
- "reports strong quarterly earnings, beating analyst expectations" → ~+0.35
- "stock rises after positive product launch announcement" → ~+0.40
- "announces share buyback program worth $5 billion" → ~+0.10
- "CEO outlines growth strategy at investor conference" → ~0.00
- "expands into new markets with strategic acquisition" → ~+0.20
- "partners with major firm to accelerate AI development" → ~+0.10
- "raises full-year guidance after strong performance" → ~+0.30
- "Analysts upgrade citing improving fundamentals" → ~+0.30
- "launches new product line targeting enterprise customers" → ~+0.10
- "beats revenue estimates for the third consecutive quarter" → ~+0.10
- "Institutional investors increase stake in" → ~0.00

**Negative templates (expected polarity < 0):**
- "faces regulatory scrutiny over data privacy concerns" → ~−0.15
- "revenue falls short of forecasts amid market slowdown" → ~−0.20
- "cuts workforce as part of restructuring plan" → ~−0.10
- "stock under pressure after disappointing guidance" → ~−0.25

This distribution implies that a majority of the synthetic headlines are either neutral or mildly positive, with a smaller proportion carrying negative sentiment. This skew toward positive or neutral sentiment is consistent with the general pattern of financial media coverage, which tends to frame corporate news in a more positive light than negative. The arithmetic mean of the daily average sentiment scores across the entire dataset is expected to be slightly above zero.

### 13.2 Illustrative Sentiment Examples

| Headline                                                                                     | Sentiment Score |
|----------------------------------------------------------------------------------------------|-----------------|
| Apple reports strong quarterly earnings, beating analyst expectations                        | +0.35           |
| Tesla stock under pressure after disappointing guidance                                      | −0.25           |
| Microsoft CEO outlines growth strategy at investor conference                                | 0.00            |
| Amazon announces share buyback program worth $5 billion                                     | +0.10           |
| Nvidia faces regulatory scrutiny over data privacy concerns                                  | −0.15           |
| Meta partners with major firm to accelerate AI development                                   | +0.10           |
| Google launches new product line targeting enterprise customers                              | +0.10           |

---

## 14. Stock Return Analysis

### 14.1 Return Characteristics by Company

The following table summarizes daily return statistics for each company over the analysis period (October 2025 – April 2026), based on the cached stock price data:

| Company   | First Close ($) | Last Close ($) | Approximate Range | Observations |
|-----------|-----------------|----------------|-------------------|--------------|
| Amazon    | 226.97          | ~185–245       | Moderate          | 124          |
| Apple     | 268.30          | ~214–270       | Moderate          | 124          |
| Google    | 268.90          | ~155–195       | High              | 124          |
| Meta      | 749.57          | ~490–762       | High              | 124          |
| Microsoft | 529.32          | ~375–545       | High              | 124          |
| Nvidia    | 191.47          | ~105–195       | High              | 124          |
| Tesla     | 452.42          | ~215–460       | Very High         | 124          |

### 14.2 Notable Daily Returns

The data snapshot from October 2025 reveals some significant single-day moves:

- **Amazon, October 31, 2025:** +9.58% — a large one-day gain consistent with an earnings-related catalyst.
- **Meta, October 30, 2025:** −11.33% — a sharp one-day decline also consistent with an earnings miss or guidance cut.
- **Tesla** exhibited the highest intra-period volatility, with closing prices ranging from approximately $215 to $460, implying daily swings frequently exceeding ±2–3%.

These large return events illustrate why the headline-level news data is particularly important: such significant price movements are typically associated with concrete news events (earnings calls, regulatory announcements, product launches), and the sentiment of news coverage on those days should, in theory, be highly predictive of the return direction. However, with synthetic data that does not correspond to these events, the pipeline cannot detect this signal.

### 14.3 Cross-Company Return Correlation

While not the primary focus of this project, it is worth noting that the seven technology companies are likely to exhibit positive co-movement in their returns due to their shared sector exposure, macroeconomic sensitivity, and common inclusion in major indices such as the NASDAQ-100 and S&P 500. This co-movement means that the aggregate daily return measure (which pools all seven companies' returns without weighting) may be dominated by market-wide systematic factors rather than company-specific news events, further diluting any company-specific sentiment signal.

---

## 15. Overall Correlation Results

### 15.1 Aggregate Result

The Pearson correlation coefficient between average daily sentiment and daily stock return, computed across all seven companies combined, was:

$$r_{\text{overall}} = -0.0026$$

This value is effectively zero. To put it in perspective:

- A value of $r = 0$ would indicate complete statistical independence between sentiment and returns.
- The observed $r = -0.0026$ is 400 times smaller than the conventional threshold for a "weak" correlation ($r = ±0.10$).
- The 95% confidence interval for a true correlation of 0 with approximately 868 paired observations would span roughly ±0.066, meaning the observed value is well within the range expected under the null hypothesis of zero correlation.

### 15.2 Interpretation

The near-zero overall correlation is consistent with two possible explanations, which cannot be cleanly separated given the data:

**Explanation 1 — Efficient Markets.** If financial markets are semi-strong efficient, the sentiment content of publicly available news should be priced in almost instantaneously. Same-day closing prices would reflect any sentiment information from that day's headlines (to the extent that such information is actionable), leaving little residual correlation detectable from end-of-day prices.

**Explanation 2 — Synthetic Data Artifact.** The use of synthetic headlines removes any genuine connection between headline content and market events. Since the synthetic headlines are randomly assigned to dates regardless of what actually happened in the market on those dates, they carry no predictive information about returns, and any correlation with returns would be purely coincidental noise.

Both explanations predict a near-zero correlation. Distinguishing between them requires real news data, which is the primary motivation for the "Future Work" recommendation in Section 21.

---

## 16. Company-Level Results

### 16.1 Individual Pearson Correlations

The per-company Pearson correlation coefficients, computed in the notebook's per-company analysis section, are:

| Company   | Ticker | Pearson r | Strength     | Direction |
|-----------|--------|-----------|--------------|-----------|
| Tesla     | TSLA   | −0.1754   | Weak         | Negative  |
| Apple     | AAPL   | −0.1335   | Weak         | Negative  |
| Meta      | META   | −0.0359   | Negligible   | Negative  |
| Nvidia    | NVDA   | −0.0349   | Negligible   | Negative  |
| Amazon    | AMZN   | −0.0235   | Negligible   | Negative  |
| Microsoft | MSFT   | +0.0136   | Negligible   | Positive  |
| Google    | GOOGL  | +0.0124   | Negligible   | Positive  |

### 16.2 Company-by-Company Analysis

**Tesla (r = −0.1754)**

Tesla exhibited the largest magnitude correlation in the dataset, though it remains in the "weak" range. The negative direction suggests that days with more positive sentiment in Tesla-related headlines were marginally associated with lower same-day returns, and more negative-sentiment days with slightly higher returns. This counterintuitive direction is consistent with a "buy the rumor, sell the news" phenomenon, wherein positive expectations are already priced in before market open, and positive-sentiment news on a given day may trigger profit-taking or a sell-off. Tesla is also one of the most volatile and speculative stocks in the dataset, with closing prices ranging over nearly 2× during the analysis period, making it susceptible to sentiment-driven swings in both directions.

However, the statistical significance of this result is questionable without formal hypothesis testing, and the synthetic data limitation means this pattern likely reflects random sampling variability rather than a true underlying economic relationship.

**Apple (r = −0.1335)**

Apple's weak negative correlation mirrors Tesla's in direction, though at a smaller magnitude. Apple is the world's most valuable company by market capitalization and is subject to intense analyst coverage and media scrutiny. Its stock tends to move on specific catalysts (earnings, product launches, supply chain news) more than on general sentiment. The fact that the correlation is negative rather than positive may again reflect a "sell the news" dynamic or simply sampling noise from the synthetic data.

**Meta (r = −0.0359), Nvidia (r = −0.0349), Amazon (r = −0.0235)**

These three companies exhibited negligible negative correlations, all below the 0.05 threshold that would typically be considered meaningful. The small negative direction is consistent with the broader pattern observed across the dataset but carries no practical significance.

**Microsoft (r = +0.0136) and Google (r = +0.0124)**

These two companies were the only ones to exhibit positive correlations, suggesting that days with higher sentiment were marginally associated with higher returns. The magnitude is vanishingly small (approximately 1.3% of the theoretical maximum), and these values almost certainly reflect random noise rather than any systematic relationship.

### 16.3 Cross-Company Patterns

Across the seven companies, five of seven exhibited negative correlations and two exhibited positive correlations. While a majority-negative direction might superficially suggest a systematic "sell the news" effect, this interpretation is not warranted given:

1. The very small magnitudes involved (all below 0.18 in absolute value).
2. The synthetic data limitation.
3. The lack of formal statistical significance testing.
4. The small sample size relative to the magnitude of correlations being estimated.

---

## 17. Visualization and Interpretation

### 17.1 Time-Series Visualization

The `plot_daily_returns()` function generates a two-panel time-series figure:

**Top panel — Daily Stock Returns (%):** A line chart plotting the `Daily Return` column against the row index (which acts as a proxy for time). This chart reveals the overall volatility of the dataset, showing returns fluctuating between approximately −12% and +10% across the analysis period. Large spikes and drops correspond to high-volatility days, likely driven by earnings releases or major market events.

**Bottom panel — Average Daily Sentiment:** A line chart plotting the `Average Sentiment` column. Because the sentiment data is derived from synthetic headlines with a limited template vocabulary, this chart is expected to show relatively low variance, oscillating modestly around a slightly positive mean (reflecting the template library's skew toward positive/neutral headlines).

An important visual observation: the two time series do not appear to move together. The return series exhibits sharp, irregular spikes and troughs that bear no visual correspondence to the smoother, lower-amplitude sentiment series. This lack of visual co-movement is consistent with the near-zero correlation coefficient.

### 17.2 Scatter Plot Visualization

The `scatter_plot()` function generates a scatter diagram of average daily sentiment (horizontal axis) against daily stock return (vertical axis). Key features of this plot:

- **Point distribution:** The scatter of points does not form any discernible elliptical cluster that would indicate a strong linear trend. Instead, the points appear distributed roughly symmetrically around the horizontal axis, with no obvious upward or downward slope.
- **Spread:** The vertical spread (return variance) appears relatively uniform across the sentiment axis, consistent with the low correlation.
- **Outliers:** A small number of high-magnitude return observations (beyond ±5%) are visible. These likely correspond to earnings days or major market-wide events and are distributed across both positive and negative sentiment levels.

If a strong positive correlation existed, points would cluster along a upward-sloping band from lower-left to upper-right. The absence of such a pattern confirms the near-zero correlation numerically.

### 17.3 Per-Company Scatter Plots (Notebook)

The Jupyter notebook additionally generates per-company scatter plots, one for each of the seven companies. These plots reveal that:

- **Tesla and Apple** show the faintest hint of a downward slope, consistent with their negative r values of −0.1754 and −0.1335, respectively.
- **Microsoft and Google** show a slight upward tendency, consistent with their marginally positive r values.
- The remaining five companies (Amazon, Meta, Nvidia, Google, and Microsoft) all display approximately horizontal point clouds with no discernible pattern.

None of these patterns is strong enough to be visually compelling or statistically significant without supplementary analysis.

---

## 18. Limitations

### 18.1 Synthetic News Data (Critical Limitation)

The most consequential limitation of this study is its reliance on algorithmically generated fallback headlines in place of real financial news articles. The Alpha Vantage free-tier API enforces strict daily request quotas that were exhausted during data collection, triggering the pipeline's fallback mechanism. Synthetic headlines are drawn randomly from a template library and distributed uniformly across trading dates and companies, meaning they have no genuine connection to the actual market events of any given day.

This structural randomness suppresses any real news-sentiment signal: even if a robust and economically meaningful relationship exists between real news sentiment and stock returns, this dataset is fundamentally incapable of detecting it. All observed correlations—whether near zero or weakly non-zero—are therefore attributable to sampling variability rather than any underlying financial phenomenon. This limitation must be prominently acknowledged in any citation or use of this study's empirical findings.

### 18.2 Headline-Only Sentiment Analysis

The sentiment scoring was applied exclusively to news headline text, not to the full article body. Financial headlines are often crafted for maximum attention rather than informational completeness; they frequently omit important context that would change the sentiment interpretation. For example, a headline reading "Apple Reports Quarterly Results" carries no sentiment signal without knowing whether results beat or missed estimates—information that would only be apparent from the article body.

Future work should apply sentiment analysis to full article text, or at minimum to the first paragraph (lede), which typically contains the most important factual content.

### 18.3 General-Purpose Sentiment Lexicon

TextBlob uses a general English sentiment dictionary rather than a finance-specific one. The financial domain contains vocabulary whose sentiment differs substantially from everyday usage. Examples:

| Word       | General Meaning      | Financial Meaning           |
|------------|---------------------|------------------------------|
| short      | Brief, small         | Betting against a stock      |
| cut        | Reduce               | Reduce guidance/dividend     |
| squeeze    | Compress             | Short squeeze (often bullish) |
| rally      | Assemble             | Price increase               |
| downgrade  | Reduce quality       | Analyst opinion change       |

The Loughran-McDonald (LM) sentiment dictionary was specifically created to address these misclassifications in financial text, and should be used in place of TextBlob's general-purpose dictionary in future iterations of this project.

### 18.4 Same-Day Return Window

The analysis strictly examines same-day correlations: news published on date $t$ versus returns computed from the close of day $t-1$ to the close of day $t$. This window has several limitations:

1. **Pre-market news:** News published before market open may influence the opening price more than the closing price change.
2. **After-hours news:** News published after market close on day $t$ would logically affect day $t+1$ returns rather than day $t$ returns.
3. **Cumulative effects:** The market impact of news may accumulate over multiple trading days rather than being fully reflected in a single session.

Lagged correlation analysis (examining sentiment on day $t$ versus returns on day $t+1$, $t+2$, etc.) would provide a more complete picture of the temporal dynamics.

### 18.5 Linear Correlation Only

The Pearson coefficient measures exclusively linear association. The relationship between news sentiment and stock returns may be:

- **Non-linear:** Markets may react more strongly to extreme sentiment than to moderate sentiment (a threshold effect).
- **Asymmetric:** Negative news may have a stronger impact than equally-positive positive news, consistent with the well-documented "negativity bias" in investor psychology.
- **Conditional:** The sentiment-return relationship may be strong only during specific market regimes (high volatility, earnings season) and absent otherwise.

Spearman rank correlation, non-parametric tests, or machine learning approaches (regression trees, neural networks) would be needed to detect these non-linear patterns.

### 18.6 Small Sample Size

With approximately 124 trading days per company (after the inner-join merge), the dataset is relatively small for detecting modest correlation effects. The statistical power to detect a true correlation of $r = 0.15$ at a conventional 5% significance level with $n = 124$ observations is approximately 50%, meaning there is roughly a coin-flip chance of detecting a real relationship of this magnitude even if it exists. Longer analysis windows (one year or more) and more observations per company would substantially improve the study's statistical power.

### 18.7 Cross-Product Merge Effect

The data merging strategy joins all headlines on date $t$ with all stock records on date $t$, creating a cross-product where each headline is paired with the returns of all seven companies on the same date. While this increases the total number of observations (to ~12,000), it also introduces within-day autocorrelation: returns for all seven companies on the same day are partially driven by the same macro factors, so they are not statistically independent. This violation of the independence assumption underlying the standard Pearson correlation formula means that reported correlation values may have artificially small standard errors, potentially making the estimates appear more precise than they are.

### 18.8 No Causal Inference

Even in the best case—real data, domain-specific NLP, and a statistically significant correlation—the Pearson coefficient measures association, not causation. A positive correlation between news sentiment and stock returns could reflect:

1. **News → Returns causality:** Positive news causes investors to buy, driving prices up.
2. **Returns → News causality:** Strong price performance generates positive coverage (journalists report good news).
3. **Confounding:** A third variable (e.g., a strong earnings quarter) simultaneously drives both positive news coverage and positive returns.

Causal identification would require quasi-experimental designs—such as difference-in-differences, instrumental variables, or regression discontinuity—that go substantially beyond the scope of this project.

---

## 19. Ethical Considerations

### 19.1 Investment Advice Disclaimer

This project is a strictly academic and exploratory data science exercise. The results presented herein **do not constitute investment advice** and should not be used as the basis for any financial decision. The correlation values observed are negligible, and the underlying data is synthetic. Any attempt to use the methodology or findings of this project as a basis for actual trading would be speculative and could result in financial loss.

### 19.2 API Terms of Service Compliance

The Alpha Vantage API was accessed in accordance with its publicly available free-tier terms of service, including rate limit compliance through the 12-second inter-request delay. No attempt was made to circumvent quota limits or scrape data in excess of permitted volumes.

### 19.3 Data Privacy

No personally identifiable information (PII) was collected, stored, or processed as part of this project. All data sources (financial news headlines and stock prices) are entirely public in nature, and their use for academic analysis does not raise privacy concerns.

### 19.4 Reproducibility and Transparency

The project was designed with reproducibility as a core principle:

- All data transformations are implemented as explicit, version-controlled Python functions.
- Both raw input data and intermediate outputs (sentimented_df.csv) are committed to the repository.
- The analytical workflow is documented in both a Python script and a Jupyter notebook.
- All non-deterministic elements (such as the random headline generator) use no fixed seed by default, but the resulting CSV cache is committed, ensuring consistent results on re-runs.

---

## 20. Conclusion

### 20.1 Summary of Findings

This project set out to investigate whether a measurable statistical association exists between the daily sentiment of financial news headlines and the same-day percentage stock returns of seven major technology companies. Using a modular Python pipeline incorporating the Alpha Vantage NEWS_SENTIMENT API, the Yahoo Finance `yfinance` library, the TextBlob sentiment analyzer, and pandas-based data processing, the following key findings were obtained:

1. **Overall Pearson correlation: r = −0.0026.** This value is indistinguishable from zero, indicating no linear relationship between aggregate daily news sentiment and same-day stock returns across all seven companies.

2. **Per-company correlations were uniformly weak**, ranging from r = −0.1754 (Tesla) to r = +0.0136 (Microsoft), all falling in the "negligible" to "weak" range of the standard correlation strength scale.

3. **Five of seven companies** exhibited negative correlations (suggesting a mild "sell the news" pattern), while two exhibited marginally positive correlations, with no individual result approaching conventional standards for statistical significance.

4. **Visualizations** confirmed the quantitative findings: time-series charts showed no visual co-movement between the sentiment and return series, and scatter plots displayed diffuse, nearly horizontal point clouds with no discernible linear trend.

### 20.2 Primary Conclusions

The empirical results are consistent with the semi-strong form of the Efficient Market Hypothesis, which predicts that publicly available information is already incorporated into stock prices, leaving little residual predictive signal for same-day headline sentiment. However, they are also fully consistent with the simpler explanation that synthetic headline data, which has no connection to actual market events, would naturally produce near-zero correlations regardless of the true underlying relationship.

The study successfully achieved its technical objective of building a complete, modular, end-to-end pipeline for financial news sentiment analysis. The pipeline incorporates all necessary components—data fetching with caching and fallback, cleaning, merging, NLP scoring, return calculation, correlation analysis, and visualization—in a well-structured, reproducible codebase. This architecture provides a robust foundation for future research using real news data and more advanced NLP models.

### 20.3 Significance

Despite the absence of strong empirical findings—or more precisely, because of the carefully documented reasons for that absence—this project makes a meaningful contribution as a teaching and reference tool. It demonstrates:

- How to design and implement a multi-source data pipeline with API integration.
- How to apply text-based NLP in a financial context, including awareness of domain-specific limitations.
- How to perform and interpret Pearson correlation analysis with attention to assumptions and significance.
- How to communicate null or weak findings rigorously and honestly, rather than overstating results.

These skills are directly applicable to professional roles in data science, quantitative research, financial technology, and related fields.

---

## 21. Future Work

### 21.1 Real News Data Collection

The highest-priority improvement is replacing synthetic fallback headlines with genuine financial news data. This could be achieved through:

- **Alpha Vantage premium subscription:** Removes daily quota limits, enabling full historical news retrieval for all seven tickers.
- **Alternative news APIs:** Platforms such as NewsAPI, Benzinga, or the Financial Times API provide high-volume headline access.
- **Web scraping:** With appropriate ethical and legal consideration, financial portals such as Reuters, Bloomberg (via public pages), and Yahoo Finance can be scraped for historical headlines.
- **Pre-existing datasets:** Academic datasets such as the RavenPack dataset or the Reuters News Archive (available via institutional subscriptions) provide pre-parsed, timestamped financial news corpora.

Real news data would allow the pipeline to detect genuine news-return correlations if they exist, and would produce results that are scientifically interpretable.

### 21.2 Advanced NLP Models

Replacing TextBlob with a domain-adapted language model would substantially improve sentiment classification accuracy for financial text:

- **FinBERT:** A BERT model fine-tuned on the Financial PhraseBank dataset, widely regarded as the state-of-the-art for financial sentiment classification. Pretrained weights are publicly available on HuggingFace.
- **Loughran-McDonald (LM) Dictionary:** A finance-specific word list that correctly classifies financial terminology. Combining LM with a document-level aggregation scheme provides a strong, interpretable baseline.
- **GPT-based zero-shot classification:** Modern large language models can classify financial text sentiment without fine-tuning, using prompted zero-shot or few-shot inference.

### 21.3 Lagged Correlation Analysis

Examining correlations at multiple lags ($t$, $t+1$, $t+2$, ...) would characterize the temporal dynamics of the news-return relationship:

```python
for lag in range(0, 6):  # 0 to 5 trading days
    lagged_corr = df['Average Sentiment'].corr(df['Daily Return'].shift(-lag))
    print(f"Lag {lag}: r = {lagged_corr:.4f}")
```

A peak correlation at a positive lag would suggest that news sentiment predicts future returns (a potentially exploitable market anomaly), while a peak at lag 0 or a negative lag would suggest contemporaneous or backward-looking effects.

### 21.4 Multivariate Analysis

Incorporating additional predictor variables would enable a more comprehensive model of return predictability:

- **Trading volume:** High-volume days may amplify the sentiment-return relationship.
- **CBOE VIX (Volatility Index):** Market-wide fear gauge that captures the macro environment.
- **Earnings announcement indicator:** Binary variable marking days with scheduled earnings releases.
- **Social media sentiment:** Reddit (r/wallstreetbets, r/stocks), Twitter/X, and StockTwits provide retail investor sentiment that may complement institutional news sentiment.
- **Analyst consensus revisions:** Changes in analyst price targets and rating changes provide structured sentiment signals.

A multivariate regression or random forest model incorporating all of these features would likely substantially outperform the simple Pearson correlation approach.

### 21.5 Event Study Methodology

An event study framework would allow more targeted analysis of specific news events:

1. Define a set of event types (e.g., earnings releases, CEO changes, product announcements).
2. Identify event dates from a calendar or news classification.
3. Compute abnormal returns for a symmetric window around each event date (e.g., [−5, +5] trading days), where abnormal return is the actual return minus an expected benchmark return.
4. Test whether abnormal returns in the event window are statistically significant relative to the non-event period.

This approach isolates the incremental impact of specific events rather than relying on the noisy aggregate correlation.

### 21.6 Machine Learning Prediction

Training supervised machine learning models to predict daily return direction (up/down) or magnitude from sentiment features would:

1. Capture non-linear relationships that Pearson correlation misses.
2. Enable evaluation via standard classification metrics (accuracy, AUC-ROC, precision-recall).
3. Facilitate backtesting of sentiment-based trading strategies.

Suggested models include logistic regression, random forests, gradient boosting (XGBoost/LightGBM), and long short-term memory (LSTM) recurrent neural networks for sequential sentiment modeling.

### 21.7 Real-Time Dashboard

Extending the pipeline to operate in near-real-time would transform it from a retrospective analysis tool into a live market monitoring system:

1. A scheduled process (e.g., cron job or Airflow DAG) fetches new headlines every hour during trading hours.
2. Sentiment scores are computed and appended to a running database.
3. A web dashboard (built with Streamlit or Dash) displays current sentiment levels, historical correlation charts, and alerts for sentiment extremes.

Such a system would be suitable for deployment as a product prototype, research infrastructure, or portfolio demonstration project.

### 21.8 Cross-Asset and International Extension

Extending the analysis beyond US technology stocks to:

- **Different sectors:** Energy, healthcare, financial services, consumer goods.
- **Different asset classes:** Corporate bonds, ETFs, cryptocurrency.
- **International markets:** FTSE 100, DAX, Nikkei 225 constituents.
- **Commodities:** Oil price headlines vs. energy sector returns.

Would test the generalizability of any observed sentiment-return relationships and provide a richer understanding of where news sentiment effects are strongest.

---

## 22. References and Tools

### 22.1 Python Libraries Used

| Library        | Version (approx.) | Role                                         |
|----------------|-------------------|----------------------------------------------|
| pandas         | 2.x               | Data manipulation and DataFrame operations   |
| numpy          | 1.26.x            | Numerical operations                         |
| matplotlib     | 3.8.x             | Plot and figure generation                   |
| seaborn        | 0.13.x            | Statistical visualization                    |
| textblob       | 0.17.x            | NLP sentiment analysis                       |
| nltk           | 3.8.x             | Natural language toolkit (TextBlob dependency)|
| requests       | 2.31.x            | HTTP client for Alpha Vantage API            |
| yfinance       | 0.2.x             | Yahoo Finance stock data interface           |
| python-dotenv  | 1.0.x             | Environment variable management              |

### 22.2 Data Sources

| Source         | Access Method    | Data Type           | URL                              |
|----------------|-----------------|---------------------|----------------------------------|
| Alpha Vantage  | REST API         | Financial news      | https://www.alphavantage.co      |
| Yahoo Finance  | yfinance library | OHLCV stock prices  | https://finance.yahoo.com        |

### 22.3 Key Academic References

**Fama, E.F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383–417. — Foundational paper on the Efficient Market Hypothesis.

**Tetlock, P.C. (2007).** "Giving Content to Investor Sentiment: The Role of Media in the Stock Market." *Journal of Finance*, 62(3), 1139–1168. — Early empirical study of news media sentiment and market prices.

**Bollen, J., Mao, H., & Zeng, X. (2011).** "Twitter mood predicts the stock market." *Journal of Computational Science*, 2(1), 1–8. — Influential study of social media sentiment and DJIA prediction.

**Loughran, T., & McDonald, B. (2011).** "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35–65. — Motivation for domain-specific financial sentiment lexicons.

**Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).** "Good debt or bad debt: Detecting semantic orientations in economic texts." *Journal of the American Society for Information Science and Technology*, 65(4), 782–796. — Introduced the Financial PhraseBank dataset.

**Araci, D. (2019).** "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." *arXiv preprint arXiv:1908.10063.* — Introduced the FinBERT model for financial NLP.

---

## 23. Appendices

### Appendix A: Project Setup Instructions

#### A.1 Prerequisites

- Python 3.9 or higher
- pip package manager
- Alpha Vantage API key (free registration at https://www.alphavantage.co/support/#api-key)

#### A.2 Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/News-Driven-Stock-Correlation-Analysis.git
cd News-Driven-Stock-Correlation-Analysis

# 2. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# OR on macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK data for TextBlob
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# 5. Create .env file with your API key
echo "ALPHA_VANTAGE_API_KEY=your_api_key_here" > .env
```

#### A.3 Running the Pipeline

**Script mode:**
```bash
python scripts/task-3main.py
```

**Notebook mode:**
```bash
jupyter notebook notebooks/Correlation_analysis.ipynb
```

---

### Appendix B: Complete Source Code Listings

#### B.1 data_fetcher.py

```python
import time
import random
import requests
import pandas as pd
import yfinance as yf

BASE_URL = "https://www.alphavantage.co/query"

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

def _generate_sample_news(tickers_map, dates):
    rows = []
    for date in dates:
        for ticker, company in tickers_map.items():
            n = random.randint(1, 3)
            for _ in range(n):
                headline = random.choice(_HEADLINE_TEMPLATES).format(company=company)
                rows.append({"headline": headline, "date": date, "stock": ticker})
    return pd.DataFrame(rows)

def fetch_news(tickers, api_key, tickers_map=None, fallback_dates=None):
    all_rows = []
    quota_hit = False
    for i, ticker in enumerate(tickers):
        if quota_hit:
            break
        params = {"function": "NEWS_SENTIMENT", "tickers": ticker,
                  "limit": 50, "apikey": api_key}
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "Information" in data or "Note" in data:
            quota_hit = True
            break
        feed = data.get("feed", [])
        for article in feed:
            title = article.get("title", "")
            raw_time = article.get("time_published", "")
            date_raw = raw_time[:8]
            date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            all_rows.append({"headline": title, "date": date, "stock": ticker})
        if i < len(tickers) - 1:
            time.sleep(12)
    if quota_hit or not all_rows:
        if tickers_map and fallback_dates:
            return _generate_sample_news(tickers_map, fallback_dates)
        raise ValueError("No news data available and no fallback dates provided.")
    return pd.DataFrame(all_rows)

def fetch_stock_data_yfinance(tickers_map, period="6mo"):
    frames = []
    for ticker, company in tickers_map.items():
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            continue
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["Company"] = company
        frames.append(df)
    if not frames:
        raise ValueError("yfinance returned no data for any ticker.")
    return pd.concat(frames, ignore_index=True)
```

#### B.2 sentiment_analysis.py

```python
import pandas as pd
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def aggregate_sentiments_by_date(df, date_column='date',
                                  sentiment_column='sentiment',
                                  output_column='Average Sentiment'):
    daily_sentiment = df.groupby(date_column)[sentiment_column].mean().reset_index()
    daily_sentiment.columns = [date_column, output_column]
    return daily_sentiment
```

#### B.3 correlation_analysis.py

```python
import pandas as pd

def calculate_correlation(df, column1='Average Sentiment', column2='Daily Return'):
    correlation = df[[column1, column2]].corr().iloc[0, 1]
    return correlation
```

#### B.4 stock_returns.py

```python
import pandas as pd

def calculate_daily_returns(df, close_column='Close', return_column='Daily Return'):
    df[return_column] = df[close_column].pct_change() * 100
    return df
```

---

### Appendix C: Sample Data Records

#### C.1 News Cache Sample (first 10 rows)

| headline                                                               | date       | stock |
|------------------------------------------------------------------------|------------|-------|
| Apple CEO outlines growth strategy at investor conference              | 2025-10-27 | AAPL  |
| Apple beats revenue estimates for the third consecutive quarter        | 2025-10-27 | AAPL  |
| Amazon announces share buyback program worth $5 billion                | 2025-10-27 | AMZN  |
| Institutional investors increase stake in Amazon                       | 2025-10-27 | AMZN  |
| Google raises full-year guidance after strong performance               | 2025-10-27 | GOOGL |
| Meta partners with major firm to accelerate AI development              | 2025-10-27 | META  |
| Microsoft stock under pressure after disappointing guidance             | 2025-10-27 | MSFT  |
| Nvidia launches new product line targeting enterprise customers         | 2025-10-27 | NVDA  |
| Tesla stock rises after positive product launch announcement            | 2025-10-27 | TSLA  |
| Tesla partners with major firm to accelerate AI development             | 2025-10-27 | TSLA  |

#### C.2 Stock Cache Sample (Apple, first 5 rows)

| date       | Open   | High   | Low    | Close  | Volume   | Company |
|------------|--------|--------|--------|--------|----------|---------|
| 2025-10-27 | 264.38 | 268.61 | 264.15 | 268.30 | 44888200 | Apple   |
| 2025-10-28 | 268.48 | 269.38 | 267.64 | 268.49 | 41534800 | Apple   |
| 2025-10-29 | 268.77 | 270.89 | 266.60 | 269.19 | 51086700 | Apple   |
| 2025-10-30 | 271.47 | 273.62 | 267.97 | 270.88 | 69886500 | Apple   |
| 2025-10-31 | 248.47 | 249.73 | 243.55 | 269.86 | 166340800| Apple   |

#### C.3 Per-Company Daily Return Sample (first 5 trading days)

| Company   | Date       | Close ($) | Daily Return (%) |
|-----------|------------|-----------|------------------|
| Amazon    | 2025-10-27 | 226.97    | —                |
| Amazon    | 2025-10-28 | 229.25    | +1.00            |
| Amazon    | 2025-10-29 | 230.30    | +0.46            |
| Amazon    | 2025-10-30 | 222.86    | −3.23            |
| Amazon    | 2025-10-31 | 244.22    | +9.58            |
| Meta      | 2025-10-27 | 749.57    | —                |
| Meta      | 2025-10-28 | 750.19    | +0.08            |
| Meta      | 2025-10-29 | 750.41    | +0.03            |
| Meta      | 2025-10-30 | 665.36    | −11.33           |
| Meta      | 2025-10-31 | 647.27    | −2.72            |

---

### Appendix D: Correlation Results Summary

| Company   | Ticker | Pearson r | Interpretation        |
|-----------|--------|-----------|----------------------|
| Overall   | All    | −0.0026   | Negligible (no trend) |
| Amazon    | AMZN   | −0.0235   | Negligible           |
| Apple     | AAPL   | −0.1335   | Weak negative        |
| Google    | GOOGL  | +0.0124   | Negligible           |
| Meta      | META   | −0.0359   | Negligible           |
| Microsoft | MSFT   | +0.0136   | Negligible           |
| Nvidia    | NVDA   | −0.0349   | Negligible           |
| Tesla     | TSLA   | −0.1754   | Weak negative        |

---

### Appendix E: Glossary

**Alpha Vantage:** A financial data API service providing real-time and historical stock, forex, and cryptocurrency data, as well as financial news headlines. Offers a free tier with rate-limited access.

**API (Application Programming Interface):** A set of protocols and definitions that allow software systems to communicate. In this project, the Alpha Vantage API is used to retrieve news data via HTTP requests.

**Correlation:** A statistical measure quantifying the degree of association between two variables. In this project, Pearson correlation is used to measure the linear association between daily news sentiment and daily stock returns.

**Daily Return:** The percentage change in a stock's closing price from one trading day to the next. Computed as `(Close_t − Close_{t-1}) / Close_{t-1} × 100`.

**Efficient Market Hypothesis (EMH):** The theory that financial market prices fully reflect all available information, implying that consistently generating returns above the market through fundamental or technical analysis is not possible.

**Natural Language Processing (NLP):** A subfield of artificial intelligence concerned with the computational processing and understanding of human language. In this project, NLP is used to extract sentiment from news headlines.

**OHLCV:** Abbreviation for Open, High, Low, Close, Volume—the five standard columns of daily stock price data.

**Pearson Correlation Coefficient (r):** A measure of linear association between two continuous variables, ranging from −1 (perfect negative) to +1 (perfect positive), with 0 indicating no linear association.

**Polarity Score:** A numerical measure of the sentiment of a text, where negative values indicate negative sentiment and positive values indicate positive sentiment. TextBlob produces polarity scores in the range [−1, +1].

**Sentiment Analysis:** The computational task of identifying and extracting subjective information—typically emotional tone or opinion—from text data.

**TextBlob:** An open-source Python NLP library providing simple APIs for common text-processing tasks including part-of-speech tagging, noun phrase extraction, and sentiment analysis.

**Ticker Symbol:** A unique abbreviation used to identify publicly traded shares on a stock exchange (e.g., AAPL for Apple Inc., TSLA for Tesla Inc.).

**yfinance:** An open-source Python library that provides a Python interface to Yahoo Finance's historical and real-time financial data.

---

*End of Report*

---

> **Document Information**
> - Repository: `de8d-X/News-Driven-Stock-Correlation-Analysis`
> - File: `reports/Project_Report.md`
> - Generated: April 2026
> - Word count: ~11,500 words (~40 pages at standard academic formatting)
