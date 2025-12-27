import numpy as np

def generateData(nData):
    x = np.random.random((10, 1))
    y = np.sin(2 * np.pi * x)

    return x, y

def initParams(order):
    w = np.random.randn(order + 1, 1)

    return w

def f(x, w):
    order = w.shape[0] - 1
    y = x ** np.arange(order + 1)
    y = y * w.T
    y = np.sum(y, axis=1, keepdims=True)
    return y


def forthProp(x, y, w):
    yHat = f(x, w)
    L = np.sum(1 / 2 * ((y - yHat) ** 2))

    return yHat, L
    
def backProp(x, y, w, yHat, L):
    order = w.shape[0] - 1

    # dL/dYHat
    dyHat = yHat - y

    # dYHat/dW = kernel
    dyHatdw = x ** np.arange(order + 1)
    # dL/dW = dL/dYHat * dYHat/dW = (y - yHat) * kernel
    dw = np.dot(dyHatdw.T, dyHat) / x.shape[0]

    return dw



def gradientDescent(learningRate, iterations):
    nData = 10
    order = 3
    x, y = generateData(nData)
    w = initParams(order)


    for i in range(1, iterations + 1):
        yHat, L = forthProp(x, y, w)
        dw = backProp(x, y, w, yHat, L)
        w = w - dw

        if i % 10000 == 0:
            print(f"Iteration #{i}: Loss = {L}")


gradientDescent(1, 50000)