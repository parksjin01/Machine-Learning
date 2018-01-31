import numpy

def gradient_descent(real_y, pred_y, feature, weight, learning_rate):

    real_y = numpy.matrix(real_y, dtype=numpy.float64)
    pred_y = numpy.matrix(pred_y, dtype=numpy.float64)
    weight = numpy.matrix(weight, dtype=numpy.float64)

    res = [0] * len(feature[0])
    res = numpy.matrix(res, dtype=numpy.float64)

    for idx in range(len(real_y)):
        res += (pred_y - real_y) * feature

    weight -= learning_rate * res.transpose()
    return weight.tolist()
