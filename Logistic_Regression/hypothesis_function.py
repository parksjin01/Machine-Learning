import numpy

def sigmoid(z):
    return 1/(1+(numpy.e)**-z)

def hypotheis(feature, weight):
    feature = numpy.matrix(feature)
    weight = numpy.matrix(weight)
    # tmp = (feature * weight).tolist()
    # print type(tmp)
    tmp = []
    for val in (feature * weight).tolist():
        tmp.append(sigmoid(val[0]))

    return tmp