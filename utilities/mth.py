class kernel:
    def __init__(self, x, order):
        self.val = []
        self.derivative = []

        for i in range(0, order + 1):
            self.val.append(pow(x, i))
            if i: self.derivative.append(i * pow(x, i - 1))
            else: self.derivative.append(0)

def dotProduct(x, y):
    assert(len(x) == len(y))
    ret = 0

    for i in range(0, len(x)):
        ret += (x[i] * y[i])

    return ret
