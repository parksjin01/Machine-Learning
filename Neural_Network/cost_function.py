"""
Cost function for logistic regression
"""

import numpy

"""
Purpose:   Calculate cost function for logistic regression
Parameter: Real output, Result of logistic regression
Result:    Cost 
"""
def cost(real_y, pred_y):
    res = 0
    for idx in range(len(real_y)):
        res += -(real_y[idx])*numpy.log(pred_y[idx])-(1-(real_y[idx]))*numpy.log(1-pred_y[idx])
    return res