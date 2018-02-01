"""
Gradient descent algorithm for logistic regression
"""

import numpy

"""
Purpose:   Optimize weight for logistic regression
Parameter: Real output, Result of logistic regression model, Feature, Weight, Learning rate
Return:    Optimized weight for logistic regression (Not perfectly optimized)
"""
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
