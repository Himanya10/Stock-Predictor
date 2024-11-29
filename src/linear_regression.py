import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import csv
from datetime import datetime
from utils import load_data, calculate_features

def write_evaluation_report(model_name, metrics):
    with open('results/evaluation_report.txt', 'a') as report:
        report.write(f"Model: {model_name}\n")
        report.write("-" * 18 + "\n")
        for metric, value in metrics.items():
            report.write(f"{metric}: {value:.2f}\n")
        report.write("\n")

def append_predictions_to_csv(date, actual, predicted, model_name):
    with open('results/predictions.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if file is empty to write the header
            writer.writerow(['Date', 'Actual_Close', 'Predicted_Close', 'Model'])
        writer.writerow([date.strftime('%Y-%m-%d'), actual, predicted, model_name])

def train_and_test_linear_regression():
    # Load and preprocess data
    data = load_data()
    data = calculate_features(data)
    
    # Features and target
    X = data[['SMA_5', 'SMA_20']]
    y = data['Close'].shift(-1).fillna(data['Close'].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Linear Regression MSE: {mse:.2f}, R^2: {r2:.2f}")
    
    # Save model
    joblib.dump(model, 'models/linear_regression_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Write evaluation report
    metrics = {
        "Mean Squared Error": mse,
        "R^2 Score": r2
    }
    write_evaluation_report("Linear Regression", metrics)

    # Append predictions to CSV
    for date, actual, pred in zip(X_test.index, y_test, predictions):
        append_predictions_to_csv(date, actual, pred, "Linear Regression")

if __name__ == "__main__":
    train_and_test_linear_regression()
