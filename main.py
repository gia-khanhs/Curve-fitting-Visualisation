from data import generateData, g
from net import network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy

#========================================================
nData = 20
xData, yData = generateData(nData)
dataFig, axis = plt.subplots()
axis.scatter(xData, yData)
#========================================================
order = 3
learningRate = 0.8 / nData #0.05
'''nData = 20, lR = 0.05'''
'''nData * lR = 0.05 * 20 = 1'''
'''=> lR = 1 / nData'''
curveFitting = network(xData, yData, order, learningRate)
#========================================================
plt.xlabel("x\n g(x) = sin(2pi * x) is the function to fit (green line). The number of data points: " + str(nData) + ". The order of the polynomial M = " + str(order))
plt.ylabel("y")

xGraph = numpy.linspace(0, 1, 40).tolist()
yGraph = []
for x in xGraph:
    yGraph.append(g(x))

xCurve = xGraph
yCurve = [0] * len(xGraph)

plt.plot(xGraph, yGraph, 'g')
curve, = plt.plot(xCurve, yCurve, 'r')
dataFig.set_size_inches(9, 6.75)
#========================================================
gdCount = [0]

def updateGraph(frame):
    curveFitting.gradientDescent(300)
    gdCount[0] += 300

    for i, x in enumerate(xCurve):
        yCurve[i] = curveFitting.polyLayer.f(x)
    
    title = "Gradient Descent Count: " + str(gdCount[0]) + "\n"
    title += "E = " + str(curveFitting.errLayer.err) + "\n"
    title += "f(x) = "
    for i, w in enumerate(reversed(curveFitting.polyLayer.w)):
        #coefficient
        if(w < 0):
            if i: title = title + "  -  "
            else: title = title + "-"
        elif i: title = title + "  +  "

        title += str(abs(round(w, 2)))

        #variable
        if(order - i):
            title += "x"
            if order - i > 1: title = title + "^" + str(order - i)
    
    plt.title(title)
    curve.set_data(xCurve, yCurve)

def artists():
    return curve

animation = FuncAnimation(dataFig, updateGraph, interval=10, init_func=artists, cache_frame_data=False)
#========================================================
plt.show()






