# %%

import numpy as np
import matplotlib.pyplot as plt


def multiplegradientDescent(
    X, y, learningRate=0.01, epochs=994, theta=None, loss_threshold=1e-3
):
    if theta is None:
        theta = np.zeros(X.shape[1])

    n = float(len(y))

    loss_y = np.zeros(epochs)

    previous_loss = float("inf")

    for i in range(epochs):
        predictedY = np.dot(X, theta)
        error = predictedY - y
        gradient = (1 / n) * np.dot(X.T, error)
        theta -= learningRate * gradient
        loss_y[i] = np.mean(error**2)

        if i > 0 and abs(loss_y[i] - previous_loss) < loss_threshold:
            print(f"stopping early at epoch {i+1} due to loss convergence.")
            actual_epochs = i + 1
            break

        previous_loss = loss_y[i]

    print(f"theta values: {theta}")

    loss_space = np.arange(1, actual_epochs + 1, 1)

    plt.plot(loss_space, loss_y[:actual_epochs])
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Loss over {actual_epochs} epochs")
    plt.show()

    final_error = np.mean((np.dot(X, theta) - y) ** 2)
    print(f"final mean squared error loss: {final_error}")

    return theta


def gradientDescent(
    x, y, independent, learningRate=0.01, epochs=1000, m=0, b=0, loss_threshold=1e-4
):
    n = float(len(x))

    loss_y = np.zeros(epochs)

    previous_loss = float("inf")

    for i in range(epochs):
        predictedY = m * x + b
        dM = (-2 / n) * np.sum(x * (y - predictedY))
        dB = (-2 / n) * np.sum(y - predictedY)
        m = m - learningRate * dM
        b = b - learningRate * dB
        totalError = np.sum((y - predictedY) ** 2) / (2 * n)
        loss_y[i] = totalError

        if i > 0 and abs(loss_y[i] - previous_loss) < loss_threshold:
            print(f"stopping early at epoch {i+1} due to loss convergence.")
            epochs = i + 1
            break
        previous_loss = loss_y[i]

    print(f"y = {m}x + {b}")

    plotData(x, y, xtitle=independent, slope=m, intercept=b)
    plt.plot(np.arange(1, epochs + 1, 1), loss_y[:epochs], "r")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Loss over {epochs} epochs")
    plt.show()

    print(f"final mean squared error loss: {totalError}")


def plotData(x, y, xtitle, slope=None, intercept=None):
    plt.scatter(x, y)

    if slope is None and intercept is None:
        return

    bestFitY = slope * x + intercept
    plt.plot(x, bestFitY, "m")
    plt.xlabel(xtitle)
    plt.ylabel("y")
    plt.title(f"{xtitle} vs y")
    plt.show()


def predictResults(independent, theta):
    ans = 1
    for x, y in zip(independent, theta[1:]):
        ans += x * y
    return ans + theta[0]


def main():
    x1 = np.loadtxt("./dataset.csv", delimiter=",", usecols=[0])
    x2 = np.loadtxt("./dataset.csv", delimiter=",", usecols=[1])
    x3 = np.loadtxt("./dataset.csv", delimiter=",", usecols=[2])
    data = np.loadtxt("./dataset.csv", delimiter=",")

    y = np.loadtxt("./dataset.csv", delimiter=",", usecols=[3])

    print("\nX1 Regression\n")
    gradientDescent(x1, y, "x1", learningRate=0.06, epochs=1000)
    print("\nX2 Regression\n")
    gradientDescent(x2, y, "x2", learningRate=0.01, epochs=5)  # nonsense data
    print("\nX3 Regression\n")
    gradientDescent(x3, y, "x3", learningRate=0.01, epochs=994)  # more nonsense data

    X = data[:, :3]
    y = data[:, 3]

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    X = np.column_stack((np.ones(X.shape[0]), X))

    theta = multiplegradientDescent(X, y, learningRate=0.05, epochs=994)
    print(f"y = {theta[1]}x1 + {theta[2]}x2 + {theta[3]}x2 + {theta[0]}")
    feature_names = ["Bias", "x1", "x2", "x3"]

    plt.bar(feature_names, theta)
    plt.xlabel("Features")
    plt.ylabel("Coefficients (Weights)")
    plt.title("Feature Importance")
    plt.show()

    # predict future results
    prediction1 = predictResults((1, 1, 1), theta)
    prediction2 = predictResults((2, 0, 4), theta)
    prediction3 = predictResults((3, 2, 1), theta)

    print(
        f"Predicted Values:\n(1,1,1): {prediction1}\n(2,0,4): {prediction2}\n(3,2,1): {prediction3}"
    )


if __name__ == "__main__":
    main()
