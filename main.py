from data import generateData, g
from net import network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy

#========================================================
nData = 5
xData, yData = generateData(nData)
# xData = [0.413400752797599, 0.6154363946085798, 0.12846952849859306, 0.663070768894383, 0.30731000979518075, 0.8780321128295719, 0.8195864936561503, 0.34484267045897943, 0.5514009011143258, 0.37707554304845825, 0.9486254158864906, 0.989794649331086, 0.10815273250521351, 0.6824563860954951, 0.17060648036791237, 0.3460869892831925, 0.9067226082556621, 0.11378613281526473, 0.04303382932056754, 0.3092527604138299]
# yData = [0.5576510447975689, -0.688743298213319, 0.7344136972680249, -0.7972044991039348, 0.8911293622858137, -0.6404636417263694, -0.8826011689245504, 0.7249840820386706, -0.29421012001645175, 0.7021995139897286, -0.2804219456548689, -0.13119274973956413, 0.8624678354025246, -0.8985535943459019, 0.859828381247477, 0.8956193647923455, -0.5380849346768445, 0.6550192941197696, 0.1538998682937392, 0.9541593070903884]
dataFig, axis = plt.subplots()
axis.scatter(xData, yData)
#========================================================
order = 10
learningRate = 0.75#0.05
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

oldE = 0

def updateGraph(frame):
    if not plt.fignum_exists(dataFig.number):
        animation.event_source.stop()
        return

    global oldE
    # if(abs(oldE - curveFitting.errLayer.err) < 0.05): return;

    nEpoch = 300
    # curveFitting.gradientDescent(nEpoch)
    # curveFitting.stochasticGD(nEpoch)
    # curveFitting.miniBatchGD(5, int(nEpoch * (nData / 5)))
    curveFitting.momentumGD(0.999, nEpoch)
    gdCount[0] += nEpoch

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
def on_close(event):
    if animation.event_source is not None:
        animation.event_source.stop()

dataFig.canvas.mpl_connect('close_event', on_close)
#========================================================
plt.show()