# %%
import numpy as np
import matplotlib.pyplot as plt

def gradientDescent(X, y, learningRate=0.01, epochs=994, theta=None, loss_threshold = 1.5e-3):
    if theta is None:
        # Initialize the weights (theta) with zeros
        theta = np.zeros(X.shape[1])
    
    n = float(len(y))


    loss_y = np.zeros(epochs)

    previous_loss = float('inf')

    for i in range(epochs):
        predictedY = np.dot(X, theta)
        error = predictedY - y
        gradient = (1/n) * np.dot(X.T, error)
        theta -= learningRate * gradient
        loss_y[i] = np.mean(error**2)

        if i > 0 and abs(loss_y[i] - previous_loss) < loss_threshold:
            print(f"stopping early at epoch {i+1} due to loss convergence.")
            actual_epochs = i+1
            break
            
        previous_loss = loss_y[i]

    print(f"theta values: {theta}")

    loss_space = np.arange(1, actual_epochs+1, 1)

    plt.plot(loss_space, loss_y[:actual_epochs])
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Loss over {actual_epochs} epochs")
    plt.show()

    final_error = np.mean((np.dot(X, theta) - y)**2)
    print(f"final mean squared error loss: {final_error}")

    return theta

def plotData(x, y, xtitle, slope=None, intercept=None):
    plt.plot(x, y)

    if slope is None and intercept is None:
        return

    bestFitY = slope * x + intercept
    plt.plot(x, bestFitY)
    plt.xlabel(xtitle)
    plt.ylabel("y")
    plt.title(f"{xtitle} vs y")
    plt.show()

def predictResults(x1,x2,x3, theta):
    return theta[0] + theta[1] * x1 + theta[2] * x2 + theta[3] * x3

def main():
    data = np.loadtxt("./dataset.csv", delimiter=',')
    
    # Separate the input features (x1, x2, x3) and the output variable (y)
    X = data[:, :3]  # Select columns 0, 1, and 2 for x1, x2, and x3
    y = data[:, 3]   # Select column 3 for the output variable

    # Standardize the input features (optional but often recommended)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # Add a bias term (intercept)
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Perform gradient descent to learn the weights
    theta = gradientDescent(X, y, learningRate=0.05, epochs=994)

    # Now you can use the learned theta to make predictions or perform further analysis
    feature_names = ["Bias", "x1", "x2", "x3"]  # Names of your features

    plt.bar(feature_names, theta)
    plt.xlabel("Features")
    plt.ylabel("Coefficients (Weights)")
    plt.title("Feature Importance")
    plt.show()

    # predict future results
    prediction1 = predictResults(1,1,1, theta)
    prediction2 = predictResults(2,0,4, theta)
    prediction3 = predictResults(3,2,1, theta)

    print(f"(1,1,1): {prediction1}\n(2,0,4): {prediction2}\n(3,2,1): {prediction3}")
    


if __name__ == "__main__":
    main()

# %%
