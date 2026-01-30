import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
import math

def evaluate_and_plot(model, X_test, y_test, scaler, csv_path, model_name='Model', color='red'):

    print(f"Generazione predizioni per {model_name}...")
    preds_scaled = model.predict(X_test)
    
    preds_scaled = preds_scaled.flatten()
    y_test = y_test.flatten()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True)
        
        start_date = '1993-01-29'
        end_date = '2023-12-28'
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df.loc[mask]
        
        LOOKBACK_WINDOW = 60
        valid_dates = df['date'].iloc[LOOKBACK_WINDOW:].values
        
        total_samples = len(valid_dates)
        val_split = int(total_samples * 0.85)
        test_dates = valid_dates[val_split:]
        
        min_len = min(len(test_dates), len(y_test))
        test_dates = test_dates[:min_len]
        y_test = y_test[:min_len]
        preds_scaled = preds_scaled[:min_len]
    else:
        raise ValueError("Colonna 'date' mancante nel CSV!")

    dummy_pred = np.zeros((len(preds_scaled), 6))
    dummy_real = np.zeros((len(y_test), 6))

    target_idx = 5 
    dummy_pred[:, target_idx] = preds_scaled
    dummy_real[:, target_idx] = y_test

    pred_dollars = scaler.inverse_transform(dummy_pred)[:, target_idx]
    real_dollars = scaler.inverse_transform(dummy_real)[:, target_idx]

    mse = mean_squared_error(real_dollars, pred_dollars)
    rmse = math.sqrt(mse)
    
    print(f"--- Risultati {model_name} ---")
    print(f"MSE: ${mse:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"Primo prezzo (Reale): ${real_dollars[0]:.2f}")
    print(f"Ultimo prezzo (Reale): ${real_dollars[-1]:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, real_dollars, label='Prezzo Reale (Adj Close)', color='blue', linewidth=1.5)
    plt.plot(test_dates, pred_dollars, label=f'Predizione {model_name}', color=color, linewidth=1.5, alpha=0.8)

    plt.title(f'{model_name}: Predizione vs Realtà (S&P 500 Test Set)', fontsize=16)
    plt.ylabel('Prezzo ($)', fontsize=12)
    plt.xlabel('Data', fontsize=12)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()