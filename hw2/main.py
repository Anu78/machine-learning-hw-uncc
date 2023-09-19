# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize(column, featureRange=(1, 5)):
   
    column = np.array(column)
    minVal = column.min()
    maxVal = column.max()

    # Scale the column values to the specified featureRange
    scaled_column = (column - minVal) * (featureRange[1] - featureRange[0]) / (maxVal - minVal) + featureRange[0]

   
    return scaled_column


def costFunction(X, y, theta):
    predictions = np.dot(X, theta)
    error = predictions - y
    cost = np.sum(error ** 2) / (2*len(y))

    return cost


def multipleDescent(X, y, numFeatures, max_iterations=1000, learningRate=0.01, tolerance=1e-5):
    theta = np.zeros(numFeatures)
    previous_cost = float('-inf')

    for iterations in range(max_iterations):
        # Calculate the gradient of the cost function
        gradient = np.dot(X.T, (np.dot(X, theta) - y)) / len(y)

        # Update parameters
        theta -= learningRate * gradient

        # Calculate the cost
        current_cost = costFunction(X, y, theta)

        # Stopping criterion
        if abs(previous_cost - current_cost) < tolerance:
            break

        previous_cost = current_cost

    return theta


def readCSV(filepath: str):
    dtypes = {
        'price': int,
        'area': int,
        'bedrooms': int,
        'bathrooms': int,
        'stories': int,
        'mainroad': str,
        'guestroom': str,
        'basement': str,
        'hotwaterheating': str,
        'airconditioning': str,
        'parking': int,
        'prefarea': str,
        'furnishingstatus': str
    }
    df = pd.read_csv(filepath, dtype=dtypes)
    df.columns = [col.strip() for col in df.columns]  # remove extra whitespace

    # data sanitization - convert yes and no to 1 and 0
    binary_columns = ['mainroad', 'guestroom', 'basement',
                      'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_columns] = df[binary_columns].replace({'yes': 1, 'no': 0})

    # convert furnishedstatus to decimal values between 0-1
    furnishing_mapping = {
        'unfurnished': 0,
        'semi-furnished': 0.5,
        'furnished': 1
    }
    df['furnishingstatus'] = df['furnishingstatus'].map(furnishing_mapping)
    normalize_columns = ['stories', 'price',
                         'area', 'bedrooms', 'bathrooms', 'parking']

    for col in normalize_columns:
        df[col] = normalize(df[col])

    # assign inputs and output
    X = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement",
            "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]]
    y = df['price']

    # split df into training and validation
    totalRows = len(df)
    trainSize = int(0.8 * totalRows)

    return (X.iloc[:trainSize], y.iloc[:trainSize]), (X.iloc[trainSize:], y.iloc[trainSize:])


def validateData(Xvalid, yvalid, theta):
    def evaluate(Xvalid, theta): return sum(
        x * t for x, t in zip(Xvalid, theta))
    columns = Xvalid.columns
    squared_errors = []
    for index in range(len(Xvalid)):
        predicted_value = evaluate(Xvalid.values[index], theta)
        error = (yvalid.values[index] - predicted_value) ** 2
        squared_errors.append(error)

        if index % 15 == 0:
            for column, value in zip(columns, Xvalid.values[index]):
                print(f"{column}: {value}", end='\t')
            print(f"price: {yvalid.values[index]}")

    mse = np.mean(squared_errors)
    return mse


def main():
    (Xtrain, ytrain), (Xvalid, yvalid) = readCSV("./Housing.csv")

    print(Xtrain, Xvalid, sep='\n\n')
    theta = multipleDescent(Xtrain, ytrain, Xtrain.shape[1], learningRate=0.078)
    feature_names = ["sq.ft", "beds", "baths", "story", "road",
                     "guest", "base", "heat", "aircon", "park", "pref", "furn"]

    plt.bar(feature_names, theta)
    plt.xlabel("Features")
    plt.ylabel("Coefficients (Weights)")
    plt.title("Feature Importance")
    plt.show()

    error = validateData(Xvalid, yvalid, theta)

    print(f"mse: {round(error,4)}")


if __name__ == "__main__":
    main()
