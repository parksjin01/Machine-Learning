"""
Gradient descent algorithm for linear regression
"""

import numpy

"""
Purpose:   Optimize weight for linear regression
Parameter: Real output, Result of linear regression model, Feature, Weight, Learning rate
Return:    Optimized weight for linear regression (Not perfectly optimized)
"""
def gradient_descent(real_y, pred_y, feature, weight, learning_rate):

    real_y = numpy.matrix(real_y, dtype=numpy.float64)
    pred_y = numpy.matrix(pred_y, dtype=numpy.float64).transpose()
    weight = numpy.matrix(weight, dtype=numpy.float64)
    feature = numpy.matrix(feature, dtype=numpy.float64)

    weight -= learning_rate * ((pred_y - real_y) * feature).transpose()
    return weight.tolist()
