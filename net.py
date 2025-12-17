from mth import kernel, dotProduct
import random
import math

sumSPG = 0

class inputLayer:
    x = []

    def __init__(self, x):
        assert(len(x))
        self.x = x

class polynomialLayer:
    #node[i] = f(x[i]) = dotProduct(kernel(x[i]), coefficients)
    
    # order = 0
    # w = [] #coefficients
    # kernel = []
    # val = []
    
    def __init__(self, iLayer, order):
        self.order = order
        self.w = [0] * (order + 1) #assume all coefficients are 0 at first
        for i in range(0, order + 1):
            self.w[i] = random.uniform(-1, 1)
        self.kernel = []
        self.val = []
        for i, x in enumerate(iLayer.x):
            self.kernel.append(kernel(x, order))
            y = dotProduct(self.kernel[-1].val, self.w)
            self.val.append(y) #because w[i] = 0

    def adjustW(self, gradient, learningRate, ada = False):
        nData = len(self.val)
        assert(len(gradient) == len(self.w))

        for i in range(len(self.w)):
            self.w[i] -= learningRate * gradient[i] / (math.sqrt(sumSPG + 0.00001) if ada else 1)
        
        for i in range(nData):
            self.val[i] = dotProduct(self.kernel[i].val, self.w)


    def f(self, x):
        k = kernel(x, self.order)
        y = dotProduct(k.val, self.w)
        return y


class errorLayer:
    # val = []
    # derivative = []

    def __init__(self, polyLayer, y):
        nData = len(y)

        self.val = []
        self.derivative = []
        self.err = 0

        for i in range(nData):
            e = polyLayer.val[i] - y[i]
            self.val.append(0.5 * e * e) #e[i] = 1/2 * (y_{guess} - y)^2
            self.derivative.append(polyLayer.val[i] - y[i])

        # self.err = math.sqrt(sum(self.val) / nData)
        self.err = sum(self.val)

class network:
    # nData = 0
    # x = []
    # y = []
    # iLayer = []
    # polyLayer = []
    # errLayer = []
    # oLayer = []

    def __init__(self, x, y, order, learningRate):
        self.nData = len(x)
        assert(len(y) == self.nData)
        self.x = x
        self.y = y
        self.learningRate = learningRate
        self.iLayer = inputLayer(x)
        self.polyLayer = polynomialLayer(self.iLayer, order)
        self.errLayer = errorLayer(self.polyLayer, y)
        self.dataID = range(self.nData)

    def getGradient(self):
        grad = [0] * (self.polyLayer.order + 1)

        for i in range(self.nData):
            dedf = self.errLayer.derivative[i] #d(err)/d(f(x))

            for j in range(self.polyLayer.order + 1):
                dfdw = self.polyLayer.kernel[i].val[j] #d(f(x))/dw
                grad[j] += dedf * dfdw #d(err)/dw

        grad = [deriv / self.nData for deriv in grad]
        global sumSPG
        sumSPG += sum([x * x for x in grad])
        return grad

    def gradientDescent(self, nEpoch = 1, ada = False):
        for i in range(nEpoch):
            g = self.getGradient()
            self.polyLayer.adjustW(g, self.learningRate, ada)
            self.errLayer = errorLayer(self.polyLayer, self.y)

    def stocGradient(self, i):
        grad = [0] * (self.polyLayer.order + 1)
        dedf = self.errLayer.derivative[i]
        for j in range(self.polyLayer.order + 1):
            dfdw = self.polyLayer.kernel[i].val[j]
            grad[j] += dedf * dfdw

        global sumSPG
        sumSPG += sum([x * x for x in grad])
        return grad

    def stochasticGD(self, nEpoch):
        for _ in range(nEpoch):
            id = [i for i in range(0, self.polyLayer.order)]
            random.shuffle(id)

            for j in id:
                g = self.stocGradient(j)
                self.polyLayer.adjustW(g, self.learningRate)
                self.errLayer = errorLayer(self.polyLayer, self.y)

    def miniBatchGradient(self, size):
        miniBatch = random.sample(self.dataID, size)
        grad = [0] * (self.polyLayer.order + 1)

        for i in miniBatch:
            dedf = self.errLayer.derivative[i]
            for j in range(self.polyLayer.order + 1):
                dfdw = self.polyLayer.kernel[i].val[j]
                grad[j] += dedf * dfdw

        grad = [deriv / size for deriv in grad]
        global sumSPG
        sumSPG += sum([x * x for x in grad])
        return grad
    
    def miniBatchGD(self, size, nEpoch = 1):
        for i in range(nEpoch):
            g = self.miniBatchGradient(size)
            self.polyLayer.adjustW(g, self.learningRate)
            self.errLayer = errorLayer(self.polyLayer, self.y)

    def momentumGD(self, gamma=0.9, nEpoch=1, ada = False):
        if not hasattr(self, "velocity"):
            self.velocity = [0] * (self.polyLayer.order + 1)

        for _ in range(nEpoch):
            grad = self.getGradient()

            for j in range(len(self.velocity)):
                self.velocity[j] = gamma * self.velocity[j] + self.learningRate * grad[j] / (math.sqrt(sumSPG + 0.01) if ada else 1)

            for j in range(len(self.polyLayer.w)):
                self.polyLayer.w[j] -= self.velocity[j]

            self.polyLayer.val = [
                dotProduct(self.polyLayer.kernel[i].val, self.polyLayer.w)
                for i in range(self.nData)
            ]
            self.errLayer = errorLayer(self.polyLayer, self.y)
