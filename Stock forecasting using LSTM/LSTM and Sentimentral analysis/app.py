from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import base64
import io
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/0.html')
def page_zero():
    return render_template('0.html')

@app.route('/5.html')
def page_five():
    return render_template('5.html')

def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        return df['Close'].rename(ticker)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return pd.Series(name=ticker)

@app.route('/predict', methods=['POST'])
def predict():
    stock = request.form['stock']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    if not start_date or not end_date:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=20*365)).strftime('%Y-%m-%d')

    data = download_data(stock, start_date, end_date)

    if data.empty:
        return "Error downloading data for the selected stock and date range."

    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    y = data.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y)

    def create_sequences(data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            target = data[i+seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    sequence_length = 10
    X_seq, y_seq = create_sequences(y_scaled, sequence_length)

    train_size, val_size = int(len(X_seq) * 0.7), int(len(X_seq) * 0.15)
    test_size = len(X_seq) - train_size - val_size
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')

    model_lstm.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    y_pred_lstm = model_lstm.predict(X_test)

    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
    y_test_inv = scaler.inverse_transform(y_test)

    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))
    rmse_rounded = round(rmse_lstm, 1)

    error_percentage = (rmse_lstm / np.mean(y_test_inv)) * 100
    error_percentage_rounded = round(error_percentage, 1)

    test_dates = data.index[train_size+val_size: train_size+val_size+len(y_test_inv)]
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label='Actual')
    plt.plot(test_dates, y_pred_lstm_inv, label='LSTM Prediction')
    plt.title(f'LSTM Prediction vs Actual (RMSE: {rmse_rounded:.1f})')
    plt.xlabel('Date')
    plt.ylabel(stock)
    plt.legend()

    # Add stock name in the top left corner
    plt.text(0.01, 0.99, stock, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    forecasted_dates = pd.date_range(test_dates[-1], periods=7, freq='D')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_data = base64.b64encode(buffer.read()).decode('utf-8')

    forecasted_values_scaled = []
    for i in range(7):
        X_new = np.array([X_test[i]])
        forecasted_value_scaled = model_lstm.predict(X_new)[0][0]
        forecasted_values_scaled.append(forecasted_value_scaled)
        X_test = np.concatenate((X_test, X_new), axis=0)

    forecasted_values = scaler.inverse_transform(np.array(forecasted_values_scaled).reshape(-1, 1))

    forecast_data_serializable = {}
    for date, price in zip(forecasted_dates, forecasted_values):
        forecast_data_serializable[date.strftime('%Y-%m-%d')] = float(price)

    return render_template('result.html', forecast=forecast_data_serializable, rmse=rmse_rounded, error_percentage=error_percentage_rounded, graph_data=graph_data)


if __name__ == '__main__':
    app.run(debug=True)