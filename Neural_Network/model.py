from Neural_Network.cost_function import *
from Neural_Network.forward_propagation import *
from Neural_Network.back_propagation import *
import numpy
feature = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
output = [[1], [0], [0], [1]]
weight1 = [[-30, 20, 20], [10, -20, -20]]
weight2 = [[-10, 20, 20]]


for idx in range(len(feature)):
    hidden = forward_propagation(numpy.matrix(feature[idx]), weight1)
    hidden.insert(0, 1)
    print hidden
    out = forward_propagation(hidden, weight2)
    delta2 = numpy.matrix(output) - numpy.matrix(out)
    delta1 = back_propagation(weight1, delta2, weight2, hidden)

    print out
    print cost(output[idx], out)
    print '\n\n\n\n\n\n'