import os
from naive_bayes import train_and_test_naive_bayes
from linear_regression import train_and_test_linear_regression
from neural_network import train_and_test_neural_network

def create_directories():
    # Create directories for saving models and results if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def main():
    # Step 1: Create directories for saving models and results
    create_directories()
    
    # Step 2: Train and evaluate models
    print("Training and testing Naive Bayes model...")
    train_and_test_naive_bayes()
    
    print("Training and testing Linear Regression model...")
    train_and_test_linear_regression()
    
    print("Training and testing Neural Network model...")
    train_and_test_neural_network()
    
    print("All models have been trained and evaluated successfully.")
    print("Predictions and evaluation reports have been saved.")

if __name__ == "__main__":
    main()
