import os
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

def load_and_process_data(ticker, start_date, end_date, na_method='drop', 
                          split_method='ratio', train_ratio=0.8, split_date=None, 
                          random_state=42, scale=False, scaler_type='standard', 
                          save_data=False, load_data=False, data_path='data.csv', 
                          scaler_path='scaler.pkl'):
    """
    Load and process stock data with the specified options.

    Parameters:
    - ticker: Stock ticker symbol.
    - start_date: Start date for the dataset.
    - end_date: End date for the dataset.
    - na_method: Method to handle NaN values ('drop' or 'fill').
    - split_method: Method to split the data ('ratio' or 'date').
    - train_ratio: Ratio of training data if split by ratio.
    - split_date: Date to split the data if split by date.
    - random_state: Random state for reproducibility.
    - scale: Whether to scale the feature columns.
    - scaler_type: Type of scaler to use ('standard' or 'minmax').
    - save_data: Whether to save the downloaded data.
    - load_data: Whether to load data from a local file.
    - data_path: Path to save/load the dataset.
    - scaler_path: Path to save/load the scaler.

    Returns:
    - X_train, X_test, y_train, y_test: Split and processed data.
    - scaler: The scaler used (if scaling is applied).
    """

    # Load data from a local file if specified
    if load_data and os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    else:
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date)
        if save_data:
            df.to_csv(data_path)

    # Handle NaN values
    if na_method == 'drop':
        df = df.dropna()
    elif na_method == 'fill':
        df = df.fillna(method='ffill').fillna(method='bfill')

    # Split the data into features and target
    X = df.drop(columns=['Adj Close'])
    y = df['Adj Close']

    # Split the data into train and test sets
    if split_method == 'ratio':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, 
                                                            random_state=random_state)
    elif split_method == 'date' and split_date:
        train_data = df[df.index < split_date]
        test_data = df[df.index >= split_date]
        X_train, y_train = train_data.drop(columns=['Adj Close']), train_data['Adj Close']
        X_test, y_test = test_data.drop(columns=['Adj Close']), test_data['Adj Close']
    else:
        raise ValueError("Invalid split method or missing split date.")

    # Scale the feature columns if specified
    scaler = None
    if scale:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type.")
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler if specified
        if save_data:
            joblib.dump(scaler, scaler_path)

    return X_train, X_test, y_train, y_test, scaler

# Example usage
X_train, X_test, y_train, y_test, scaler = load_and_process_data(
    ticker="AAPL",
    start_date="2018-12-17",
    end_date="2019-12-17",
    na_method='drop',
    split_method='ratio',
    train_ratio=0.8,
    scale=True,
    scaler_type='standard',
    save_data=True,
    load_data=False,
    data_path='AAPL_data.csv',
    scaler_path='AAPL_scaler.pkl'
)
