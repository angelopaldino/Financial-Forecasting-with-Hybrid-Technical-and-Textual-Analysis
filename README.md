# Financial Forecasting with Hybrid Technical and Textual Analysis
<img src = "https://private-user-images.githubusercontent.com/175546677/542791947-81547740-bc37-458b-810a-cea68aef00be.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk3NzE0NzgsIm5iZiI6MTc2OTc3MTE3OCwicGF0aCI6Ii8xNzU1NDY2NzcvNTQyNzkxOTQ3LTgxNTQ3NzQwLWJjMzctNDU4Yi04MTBhLWNlYTY4YWVmMDBiZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEzMFQxMTA2MThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00ZTA1ZDRlZWFhNGE0OTRjNzZmNWRhMGFjNzA0ZTZjMGFmYWVjYWIwMzk2YjViOTVkNWQ4ZGFiMzY1ZGY0NTU3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hMNUkpHwdg2q0yQfzIZ2uC_Ff3GgiFOzNa5lP-u0jts" width = 800>
## Introduction

This project addresses the challenge of financial time series forecasting by proposing a hybrid methodology that integrates quantitative market data with qualitative textual information.

Building upon the foundations laid in the research paper **"[Insert Original Paper Title Here]"**, this work extends the original scope by incorporating advanced Natural Language Processing (NLP) techniques. While traditional approaches often rely solely on historical price and volume data, our research hypothesizes that financial news and market sentiment play a crucial role in predicting short-term market movements.

The core innovation of this repository is the enrichment of the input dataset with sentiment scores derived from **FinBERT**, a BERT model specifically fine-tuned for the financial domain, combined with Deep Learning architectures (LSTM and CNN) to capture both temporal dependencies and sentiment-driven volatility.

## Project Overview

The primary objective is to predict the future trend or price of a financial asset (e.g., SPY, individual stocks) using a multivariate dataset.

### The Hybrid Approach
Financial markets are influenced by two main factors:
1.  **Technical Factors:** Historical price trends, volume, and volatility.
2.  **Fundamental/Psychological Factors:** News, earnings reports, and general investor sentiment.

This project merges these two streams. We utilize historical market data as the technical baseline and augment it with a "Sentiment Score" feature. This score is not a simple dictionary-based count but a semantic representation extracted using FinBERT, allowing the model to understand the context of financial news headlines.

## Methodology and Architecture

The pipeline consists of three main stages: Data Preprocessing, Model Training, and Explainability.

### 1. Feature Engineering and NLP
* **Technical Data:** Standard OHLCV (Open, High, Low, Close, Volume) data is normalized and structured into sliding windows (timesteps) to serve as input for sequential models.
* **Textual Data (Sentiment Analysis):**
    * Financial news headlines associated with the target asset were collected.
    * We employed **FinBERT** (Financial Bidirectional Encoder Representations from Transformers) to analyze the sentiment of each headline.
    * The model outputs a probability distribution (Positive, Negative, Neutral), which is converted into a numerical `sentiment_score`.
    * This score is aligned temporally with the market data, creating a unified multivariate time series.

### 2. Predictive Models
We implemented and compared several architectures to evaluate the effectiveness of the hybrid dataset:

* **LSTM (Long Short-Term Memory):** Chosen for its ability to learn long-term dependencies in time series data. The LSTM network receives sequences of technical indicators and sentiment scores to predict the next timestep's value.
* **CNN (Convolutional Neural Networks):** Adapted for time series to extract local invariant features and patterns from the windowed data.
* **Decision Tree:** Used as a baseline model to compare performance against deep learning approaches and to provide inherent interpretability.

### 3. Explainable AI (XAI)
A critical component of this project is understanding *how* the models make decisions, overcoming the "black box" nature of Deep Learning. We applied three state-of-the-art XAI techniques:

* **SHAP (SHapley Additive exPlanations):** Used to determine the global importance of features. SHAP analysis helped us verify that the model relies heavily on price history (Close, Open) while utilizing Sentiment as a correction factor for volatility.
* **LIME (Local Interpretable Model-agnostic Explanations):** Used to explain individual predictions. This allowed us to observe specific instances where the model's decision was influenced by a sudden shift in sentiment versus technical trends.
*  **PERMUTANCE IMPORTANCE: used to perturb each feature one at a time in order to visualize which feature was most important.

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # Original market and news data
│   └── processed/           # Normalized numpy arrays (X_train, y_train, etc.)
├── models/
│   ├── hybrid_models/       # Saved Keras (.keras) and Joblib (.pkl) models
│   └── baseline_utils/      
├── notebooks/               # Jupyter notebooks
├── utils/                   # helper function to display graphs
└── README.md                # This document
