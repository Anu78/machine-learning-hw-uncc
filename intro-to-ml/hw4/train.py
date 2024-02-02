#! /usr/bin/python3.11

import numpy as np
import pandas as pd
from models import SVM, SVR
from metrics import Metrics


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


def pca(data, k):
    # Calculate the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Ensure k is within the valid range
    k = min(data.shape[1], k)

    # Select the top k eigenvectors
    projection_matrix = eigenvectors[:, :k]

    # Project the data onto the selected eigenvectors
    data_pca = np.dot(data, projection_matrix)

    return pd.DataFrame(data_pca)


if __name__ == "__main__":
    #! SVM training on cancer dataset
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
    replace = {"B": 1, "M": -1}
    df, outcome = readCSV(
        "./data/cancer.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )
    df_valid, outcome_valid = readCSV(
        "./data/cancer-valid.csv",
        scaler=scaler,
        dtypes=dtypes,
        output="diagnosis",
        replace=replace,
        remove=["id"],
    )

    # implement pca feature extraction
    k = 15  # amount of features to keep
    df_pca = pca(df, k)
    df_valid_PCA = pca(df_valid, k)

    # train!
    model = SVM()

    model.fit(df.values, outcome.values)

    # validate accuracy
    model_out = model.predict(df_valid.values)

    # get metrics
    metrics = Metrics(model_out, outcome_valid)
    print(
        f"accuracy: {metrics.acc}, precision: {metrics.precision}, recall: {metrics.recall}, f1: {metrics.f1}"
    )
    print(metrics.matrix)

    #! SVR training on housing dataset
    df = pd.read_csv("./data/Housing.csv")
    outcome = df.pop("price")
    df_valid = pd.read_csv("./data/Housing-valid.csv")
    outcome_valid = df_valid.pop("price")

    replacement_dict = {"yes": 1, "no": 0}

    # Columns to replace values in
    columns_to_replace = [
        "mainroad",
        "guestroom",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "basement",
    ]

    # Use the replace method to perform the replacements
    df[columns_to_replace] = df[columns_to_replace].replace(replacement_dict)
    df_valid[columns_to_replace] = df_valid[columns_to_replace].replace(
        replacement_dict
    )

    replacement_dict = {"furnished": 1, "semi-furnished": 0.5, "unfurnished": 0}
    df["furnishingstatus"] = df["furnishingstatus"].replace(replacement_dict)
    df_valid["furnishingstatus"] = df_valid["furnishingstatus"].replace(
        replacement_dict
    )

    # normalize data now
    def min_max_scaling(col):
        if col.dtype in [int, float]:
            minval = col.min()
            maxval = col.max()

            scaledcol = (col - minval) / (maxval - minval)

            return scaledcol
        else:
            return col

    # normalize inputs and outputs
    df = df.apply(min_max_scaling)
    df_valid = df_valid.apply(min_max_scaling)
    outcome = min_max_scaling(outcome)
    outcome_valid = min_max_scaling(outcome_valid)

    # apply PCA
    k = 5
    df_pca = pca(df, k)

    # train!
    model = SVR()
    model.fit(df.values, outcome.values)
    print(model.alpha, model.b)

    # validate data
    model_out = model.predict(df_valid.values)
    # accuracy
    metrics = Metrics(model_out, outcome_valid.values)
    print(
        f"accuracy: {metrics.acc}, precision: {metrics.precision}, recall: {metrics.recall}, f1: {metrics.f1}"
    )
    print(metrics.matrix)
