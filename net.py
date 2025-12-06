from mth import kernel, crossProduct

class inputLayer:
    x = []

    def __init__(self, x):
        assert(len(x))
        self.x = x

class polynomialLayer:
    #node[i] = f(x[i]) = crossProduct(kernel(x[i]), coefficients)
    
    # order = 0
    # w = [] #coefficients
    # kernel = []
    # val = []
    
    def __init__(self, iLayer, order):
        self.order = order
        self.w = [0] * (order + 1) #assume all coefficients are 0 at first
        self.kernel = []
        self.val = []
        for i, x in enumerate(iLayer.x):
            self.kernel.append(kernel(x, order))
            self.val.append(0) #because w[i] = 0

    def adjustW(self, gradient, learningRate):
        nData = len(self.val)
        assert(len(gradient) == len(self.w))

        for i in range(len(self.w)):
            self.w[i] -= learningRate * gradient[i]
        
        for i in range(nData):
            self.val[i] = crossProduct(self.kernel[i].val, self.w)

    def f(self, x):
        k = kernel(x, self.order)
        y = crossProduct(k.val, self.w)
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

    def getGradient(self):
        grad = [0] * (self.polyLayer.order + 1)

        for i in range(self.nData):
            dedf = self.errLayer.derivative[i] #d(err)/d(f(x))

            for j in range(self.polyLayer.order + 1):
                dfdw = self.polyLayer.kernel[i].val[j] #d(f(x))/dw
                grad[j] += dedf * dfdw #d(err)/dw

        return grad

    def gradientDescent(self, nEpoch = 1):
        for i in range(nEpoch):
            g = self.getGradient()
            self.polyLayer.adjustW(g, self.learningRate)
            self.errLayer = errorLayer(self.polyLayer, self.y)
