# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readCSV(filepath, scaler="nrml"):
    dtypes = {
        'Pregnancies': int,
        'Glucose': int,
        'BloodPressure': int,
        'SkinThickness': int,
        'Insulin': int,
        'BMI': float,
        'DiabetesPedigreeFunction': float,
        'Age': int,
        'Outcome': int
    }
    df = pd.read_csv(filepath, dtype=dtypes)

    outcome = df.pop("Outcome")

    if scaler == "normalize":
        df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        means = df.mean()
        stds = df.std()
        df = (df - means) / stds

    return df, outcome

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta, bias):
    linear_model = np.dot(X, theta) + bias
    return sigmoid(linear_model)

def evaluate_accuracy(X, y, theta, bias):
    # Make predictions on the validation data
    y_pred = predict(X, theta, bias)
    
    # Threshold the predictions (e.g., >= 0.5 as 1, < 0.5 as 0)
    y_pred = np.round(y_pred)
    
    # Calculate accuracy
    correct_predictions = np.sum(y_pred == y)
    total_predictions = len(y)
    accuracy = correct_predictions / total_predictions
    
    return accuracy

def logisticRegression(X, y, learningRate=0.1, iterations = 300):
    num_samples, num_features = X.shape
    
    theta = np.zeros(num_features)
    bias = 0

    for i in range(iterations):
        model = np.dot(X, theta) + bias
        predictions = sigmoid(model)

        dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
        db = (1 / num_samples) * np.sum(predictions - y)

        theta -= learningRate * dw
        bias -= learningRate * db
    
    return theta, bias

def plotFeatures(featureNames, theta):
    plt.bar(featureNames, theta)
    plt.xlabel("Features")
    plt.ylabel("Coefficients (Weights)")
    plt.title("Feature Importance")
    plt.show()

def main():
    df, outcome = readCSV("./diabetes.csv", scaler="std")
    df_valid, outcome_valid = readCSV("./diabetes-valid.csv")

    theta, bias = logisticRegression(df, outcome, learningRate=0.01)
    accuracy = evaluate_accuracy(df_valid, outcome_valid, theta, bias)
    
    print(theta, bias, f"{round(accuracy*100,4)}% accuracy")

    features = ["preg", "glucose", "bp", "skintick", "insulin", "bmi", "dpf", "age"]

    plotFeatures(features, theta)



if __name__ == "__main__":
    main()
