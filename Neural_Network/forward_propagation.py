import numpy

def sigmoid(z):
    res = []
    for num in z:
        res.append(1/(1+(numpy.e)**-num[0]))
    return res


def forward_propagation(feature, weight):
    weight = numpy.matrix(weight, numpy.float64)
    feature = numpy.matrix(feature, numpy.float64)

    return sigmoid((weight * feature.transpose()).tolist())