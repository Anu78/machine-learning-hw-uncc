# %%
# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(x, y, independent, learningRate=0.01, epochs=1000, m=0, b=0, loss_threshold=1e-4):

    n = float(len(x))

    loss_y = np.zeros(epochs)

    previous_loss = float('inf')

    for i in range(epochs):
        predictedY = m * x + b
        dM = (-2/n) * np.sum(x * (y - predictedY))
        dB = (-2/n) * np.sum(y - predictedY)
        m = m - learningRate * dM
        b = b - learningRate * dB
        totalError = np.sum((y - predictedY)**2) / (2 * n)
        loss_y[i] = totalError

        if i > 0 and abs(loss_y[i] - previous_loss) < loss_threshold:
            print(f"stopping early at epoch {i+1} due to loss convergence.")
            epochs = i+1
            break
        previous_loss = loss_y[i]

    print(f"y = {m}x + {b}")

    plotData(x, y, xtitle=independent, slope=m, intercept=b)
    plt.plot(np.arange(1, epochs+1, 1), loss_y[:epochs], 'r')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Loss over {epochs} epochs")
    plt.show()

    print(f"final mean squared error loss: {totalError}")


def multiGradientDescent():
    pass


def plotData(x, y, xtitle, slope=None, intercept=None):
    plt.scatter(x, y)

    if slope is None and intercept is None:
        return

    bestFitY = slope * x + intercept
    plt.plot(x, bestFitY, 'm')
    plt.xlabel(xtitle)
    plt.ylabel("y")
    plt.title(f"{xtitle} vs y")
    plt.show()


def main():
    x1 = np.loadtxt("./dataset.csv", delimiter=',', usecols=[0])
    x2 = np.loadtxt("./dataset.csv", delimiter=',', usecols=[1])
    x3 = np.loadtxt("./dataset.csv", delimiter=',', usecols=[2])

    y = np.loadtxt("./dataset.csv", delimiter=',', usecols=[3])

    gradientDescent(x1, y, "x1", learningRate=0.005, epochs=1000)

    # gradientDescent(x2,y, "x2", learningRate=0.01, epochs=5) # nonsense data

    # gradientDescent(x3,y, "x3", learningRate=0.01, epochs=994) # more nonsense data


if __name__ == "__main__":
    main()

# %%
