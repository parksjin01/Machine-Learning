import numpy

def hypotheis(feature, weight):
    feature = numpy.matrix(feature)
    weight = numpy.matrix(weight)
    # tmp = (feature * weight).tolist()
    # print type(tmp)

    return (feature * weight).tolist()