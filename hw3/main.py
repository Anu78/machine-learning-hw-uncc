# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readCSV(filepath, dtypes, output, scaler="", replace=None, remove=[]):
    df = pd.read_csv(filepath, dtype=dtypes)
    outcome = df.pop(output)
    if scaler == "nrml":
        df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif scaler == "std":
        means = df.mean()
        stds = df.std()
        df = (df - means) / stds

    if replace is not None:
        outcome = outcome.replace(replace)
    if remove:
        for col in remove:
            del df[col]

    return df, outcome


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, theta, bias):
    linear_model = np.dot(X, theta) + bias
    return sigmoid(linear_model)


def logisticRegression(
    X, y, learningRate=0.1, iterations=1000, tolerance=1e-4, strength=0.1, reg=""
):
    num_samples, num_features = X.shape

    theta = np.zeros(num_features)
    bias = 0
    previous_cost = float("inf")
    costs = []

    for i in range(iterations):
        model = np.dot(X, theta) + bias
        predictions = sigmoid(model)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

        cost = (
            -1
            / num_samples
            * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        )

        if reg == "l1":
            cost += strength * np.sum(np.abs(theta))
        elif reg == "l2":
            cost += strength * np.sum(np.square(theta))

        costs.append(cost)

        dw = (1 / num_samples) * np.dot(X.T, (predictions - y))

        if reg == "l1":
            dw += strength * np.sign(theta)
        elif reg == "l2":
            dw += 2 * strength * theta
        db = (1 / num_samples) * np.sum(predictions - y)

        theta -= learningRate * dw
        bias -= learningRate * db

        if i > 0 and abs(previous_cost - cost) < tolerance:
            break

        previous_cost = cost

    plt.plot(np.arange(i + 1), costs)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title(f"Loss over {i} epochs")
    plt.show()

    return theta, bias


def plotFeatures(featureNames, theta):
    plt.bar(featureNames, theta)
    plt.xlabel("Features")
    plt.ylabel("Coefficients (Weights)")
    plt.title("Feature Importance")
    plt.show()


def accuracy_(predicted, y):
    correct_predictions = np.sum(predicted == y)
    total_predictions = len(y)
    accuracy = correct_predictions / total_predictions
    return accuracy


def precision_(matrix):
    tp = matrix[0][0]
    fp = matrix[1][0]

    return round(tp / (tp + fp), 4)


def recall_(matrix):
    tp = matrix[0][0]
    fn = matrix[1][1]

    return round(tp / (tp + fn), 4)


def f1_(precision, recall):
    return round((2 * precision * recall) / (precision + recall), 4)


def confusionMatrix(predicted, y):
    tp = np.sum(predicted == y)
    tn = np.sum(predicted != y)
    fp = sum(
        1
        for true_label, predicted_label in zip(y, predicted)
        if predicted_label == 1 and true_label == 0
    )
    fn = sum(
        1
        for true_label, predicted_label in zip(y, predicted)
        if predicted_label == 0 and true_label == 1
    )

    return [[tp, tn], [fp, fn]]


class NaiveBayesClassifier:
    def __init__(self):
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)
        for label in unique_classes:
            X_class = X[y == label]
            self.mean[label] = X_class.mean(axis=0)
            self.var[label] = X_class.var(axis=0)
            self.priors[label] = float(len(X_class) / len(X))

    def predict(self, X):
        probs = np.zeros((len(X), len(self.priors)))
        for label, prior in self.priors.items():
            probs[:, label] = (
                np.log(prior)
                + -0.5 * np.sum(np.log(2 * np.pi * self.var[label]))
                - 0.5
                * np.sum(((X - self.mean[label]) ** 2) / (self.var[label]), axis=1)
            )
        return np.argmax(probs, axis=1)


def pca(data, k):
    covariance_matrix = np.cov(data, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    k = min(data.shape[1], k)

    projection_matrix = eigenvectors[:, :k]

    data_pca = np.dot(data, projection_matrix)

    return pd.DataFrame(data_pca)


def problem1():
    scaler = "std"
    dtypes = {
        "Pregnancies": int,
        "Glucose": int,
        "BloodPressure": int,
        "SkinThickness": int,
        "Insulin": int,
        "BMI": float,
        "DiabetesPedigreeFunction": float,
        "Age": int,
        "Outcome": int,
    }
    df, outcome = readCSV(
        "./datasets/diabetes.csv", scaler=scaler, dtypes=dtypes, output="Outcome"
    )
    df_valid, outcome_valid = readCSV(
        "./datasets/diabetes-valid.csv", scaler=scaler, dtypes=dtypes, output="Outcome"
    )

    theta, bias = logisticRegression(
        df, outcome, learningRate=0.01, strength=0.1, reg=""
    )

    predicted = predict(df_valid, theta, bias)
    predicted = np.round(predicted)

    confMatrix = confusionMatrix(predicted, outcome_valid)

    print("\nConfusion Matrix:")
    for row in confMatrix:
        print(row)

    accuracy = accuracy_(predicted, outcome_valid)
    precision = precision_(confMatrix)
    recall = recall_(confMatrix)
    f1 = f1_(precision, recall)

    print(f"\nAccuracy: {round(accuracy*100,4)}%")
    print(f"Precision: {precision*100}%")
    print(f"Recall: {recall*100}%")
    print(f"F1: {f1}")

    plotFeatures(
        ["preg", "glucose", "bp", "skinthick", "insulin", "bmi", "dpf", "age"], theta
    )


def problem2():
    scaler = "nrml"
    dtypes = {
        "id": int,
        "diagnosis": str,
        "radius_mean": float,
        "texture_mean": float,
        "perimeter_mean": float,
        "area_mean": float,
        "smoothness_mean": float,
        "compactness_mean": float,
        "concavity_mean": float,
        "concave points_mean": float,
        "symmetry_mean": float,
        "fractal_dimension_mean": float,
        "radius_se": float,
        "texture_se": float,
        "perimeter_se": float,
        "area_se": float,
        "smoothness_se": float,
        "compactness_se": float,
        "concavity_se": float,
        "concave points_se": float,
        "symmetry_se": float,
        "fractal_dimension_se": float,
        "radius_worst": float,
        "texture_worst": float,
        "perimeter_worst": float,
        "area_worst": float,
        "smoothness_worst": float,
        "compactness_worst": float,
        "concavity_worst": float,
        "concave points_worst": float,
        "symmetry_worst": float,
        "fractal_dimension_worst": float,
    }
    replace = {"B": 1, "M": 0}
    df, outcome = readCSV(
        "./datasets/cancer.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )
    df_valid, outcome_valid = readCSV(
        "./datasets/cancer-valid.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )

    theta, bias = logisticRegression(
        df, outcome, learningRate=0.1, strength=0.01, reg="l1"
    )

    predicted = predict(df_valid, theta, bias)
    predicted = np.round(predicted)

    confMatrix = confusionMatrix(predicted, outcome_valid)

    print("\nConfusion Matrix:")
    for row in confMatrix:
        print(row)

    accuracy = accuracy_(predicted, outcome_valid)
    precision = precision_(confMatrix)
    recall = recall_(confMatrix)
    f1 = f1_(precision, recall)

    print(f"\nAccuracy: {round(accuracy*100,4)}%")
    print(f"Precision: {precision*100}%")
    print(f"Recall: {round(recall*100,4)}%")
    print(f"F1: {f1}")


def problem3():
    scaler = "nrml"
    dtypes = {
        "id": int,
        "diagnosis": str,
        "radius_mean": float,
        "texture_mean": float,
        "perimeter_mean": float,
        "area_mean": float,
        "smoothness_mean": float,
        "compactness_mean": float,
        "concavity_mean": float,
        "concave points_mean": float,
        "symmetry_mean": float,
        "fractal_dimension_mean": float,
        "radius_se": float,
        "texture_se": float,
        "perimeter_se": float,
        "area_se": float,
        "smoothness_se": float,
        "compactness_se": float,
        "concavity_se": float,
        "concave points_se": float,
        "symmetry_se": float,
        "fractal_dimension_se": float,
        "radius_worst": float,
        "texture_worst": float,
        "perimeter_worst": float,
        "area_worst": float,
        "smoothness_worst": float,
        "compactness_worst": float,
        "concavity_worst": float,
        "concave points_worst": float,
        "symmetry_worst": float,
        "fractal_dimension_worst": float,
    }
    replace = {"B": 1, "M": 0}
    df, outcome = readCSV(
        "./datasets/cancer.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )
    df_valid, outcome_valid = readCSV(
        "./datasets/cancer-valid.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )

    clf = NaiveBayesClassifier()
    clf.fit(df, outcome)

    predictions = clf.predict(df_valid)

    confMatrix = confusionMatrix(predictions, outcome_valid)

    print("\nConfusion Matrix:")
    for row in confMatrix:
        print(row)

    accuracy = accuracy_(predictions, outcome_valid)
    precision = precision_(confMatrix)
    recall = recall_(confMatrix)
    f1 = f1_(precision, recall)

    print(f"\nAccuracy: {round(accuracy*100,4)}%")
    print(f"Precision: {precision*100}%")
    print(f"Recall: {round(recall*100,4)}%")
    print(f"F1: {f1}")


def problem4(k=5, logistic=True):
    scaler = "nrml"
    dtypes = {
        "id": int,
        "diagnosis": str,
        "radius_mean": float,
        "texture_mean": float,
        "perimeter_mean": float,
        "area_mean": float,
        "smoothness_mean": float,
        "compactness_mean": float,
        "concavity_mean": float,
        "concave points_mean": float,
        "symmetry_mean": float,
        "fractal_dimension_mean": float,
        "radius_se": float,
        "texture_se": float,
        "perimeter_se": float,
        "area_se": float,
        "smoothness_se": float,
        "compactness_se": float,
        "concavity_se": float,
        "concave points_se": float,
        "symmetry_se": float,
        "fractal_dimension_se": float,
        "radius_worst": float,
        "texture_worst": float,
        "perimeter_worst": float,
        "area_worst": float,
        "smoothness_worst": float,
        "compactness_worst": float,
        "concavity_worst": float,
        "concave points_worst": float,
        "symmetry_worst": float,
        "fractal_dimension_worst": float,
    }
    replace = {"B": 1, "M": 0}
    df, outcome = readCSV(
        "./datasets/cancer.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )
    df_valid, outcome_valid = readCSV(
        "./datasets/cancer-valid.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )

    df = pca(df, k)
    df_valid = pca(df_valid, k)

    if logistic:
        theta, bias = logisticRegression(
            df, outcome, learningRate=0.01, strength=0.1, reg=""
        )

        predicted = predict(df_valid, theta, bias)
        predicted = np.round(predicted)
    else:
        clf = NaiveBayesClassifier()
        clf.fit(df, outcome)

        predicted = clf.predict(df_valid)

    confMatrix = confusionMatrix(predicted, outcome_valid)

    print("\nConfusion Matrix:")
    for row in confMatrix:
        print(row)

    accuracy = accuracy_(predicted, outcome_valid)
    precision = precision_(confMatrix)
    recall = recall_(confMatrix)
    f1 = f1_(precision, recall)

    print(f"\nAccuracy: {round(accuracy*100,4)}%")
    print(f"Precision: {precision*100}%")
    print(f"Recall: {recall*100}%")
    print(f"F1: {f1}")


def problem5():
    pass


def main():
    problem1()

    problem2()

    problem3()

    problem4(k=10, logistic=True)

    problem4(k=10, logistic=False)


if __name__ == "__main__":
    main()
