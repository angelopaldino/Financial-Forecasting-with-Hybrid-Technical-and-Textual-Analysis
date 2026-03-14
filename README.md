# Financial Forecasting & Multimodal XAI
This repository presents an end-to-end framework for financial time series (S&P 500) forecasting by integrating quantitative analysis and qualitative sentiment analysis. The central focus is the critical investigation of model transparency using Explainable AI (XAI) techniques.
Project based on the paper: "Evaluating Transparency: A Cross-Model Exploration of Explainable AI in Financial Forecasting."
## Objectives of the Project
Multimodal Merger: Integrate technical data (prices/volumes) with textual data (financial news). Specialist Sentiment: Use FinBERT for accurate semantic decoding of economic language. Architectural Benchmarking: Compare Recurrent (LSTM), Convolutional (CNN), and Tree (Decision Tree) models. "Black-Box" Decoding: Apply Permutation Importance, SHAP, LIME (XAI) techniques to understand the decision logics of complex models.
## Dataset: FNSPID
The project uses the Financial News and Stock Price Integration Dataset (FNSPID):

Scale: 29.7M price records and 15.7M financial news.

Coverage: 4,775 companies (S&P 500) from 1999 to 2023.

Target: SPY ETF (S&P 500) as a global liquidity barometer.

## Development Pipeline
1. Sentiment Extraction (NLP) Using FinBERT (ProsusAI/finbert) to transform news into probabilistic vectors (Positive, Negative, Neutral).Input: News Headlines.Output: Continuous scores in ranges $[-1, 1]$ that shape market emotion.
2. Multimodal Data Engineering Synchronization: Temporal alignment between discrete technical data and asynchronous news. Sliding Window: Creating 3D tensors with a 60-day lookback window. Data Split: Sequential division (Training 70%, Val 15%, Test 15%) to avoid data leakage.
3. XAI


