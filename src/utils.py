import pandas as pd

def load_data():
    """Loads the stock prices dataset."""
    return pd.read_csv('data/stock_prices.csv')

def calculate_features(data):
    """Calculates additional features for the dataset."""
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data.dropna(inplace=True)
    return data
