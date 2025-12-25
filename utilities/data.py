import numpy
import cmath
import random

def g(x):
    return cmath.sin(2 * cmath.pi * x)

def generateData(N = 20, l = 0.0, r = 1):
    stdDev = 0.1
    retX = []
    retY = []
    for i in range(N):
        x = random.uniform(l, r)
        mean = g(x)
        y = random.gauss(mean, stdDev).real

        retX.append(x)
        retY.append(y)
    
    return (retX, retY)