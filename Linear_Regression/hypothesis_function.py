"""
Hypothesis function for linear regression
"""

import numpy

"""
Purpose:   Linear regression model's hypothesis function
Parameter: Feature, Weight
Return:    Result of hypothesis function
"""
def hypotheis(feature, weight):
    feature = numpy.matrix(feature)
    weight = numpy.matrix(weight)

    return (feature * weight).tolist()