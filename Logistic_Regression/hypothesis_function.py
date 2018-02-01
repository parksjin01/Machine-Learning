"""
Hypothesis function for logistic regression
"""

import numpy

"""
Purpose:   Sigmoid function which is used in logistic regression hypothesis function
Parameter: weight * feature
Return:    Result of sigmoid function
"""
def sigmoid(z):
    return 1/(1+(numpy.e)**-z)

"""
Purpose:   Logistic regression model's hypothesis function
Parameter: Feature, Weight
Return:    Result of hypothesis function
"""
def hypotheis(feature, weight):
    feature = numpy.matrix(feature)
    weight = numpy.matrix(weight)
    tmp = []
    for val in (feature * weight).tolist():
        tmp.append(sigmoid(val[0]))

    return tmp