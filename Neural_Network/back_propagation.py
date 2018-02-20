import numpy

def deriv_sigmoid(z):
    z = numpy.matrix(z, numpy.float64)
    return numpy.multiply(z, 1-z)

def back_propagation(weight, prev_delta, prev_weight, prev_feature):
    weight = numpy.matrix(weight, numpy.float64)
    prev_delta = numpy.matrix(prev_delta, numpy.float64)
    prev_weight = numpy.matrix(prev_weight, numpy.float64)
    prev_feature = numpy.matrix(prev_feature, numpy.float64)

    return numpy.multiply(weight.transpose() * prev_delta, deriv_sigmoid(prev_weight * prev_feature))
