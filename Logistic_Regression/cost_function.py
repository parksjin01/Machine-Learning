import numpy

def cost(real_y, pred_y):
    res = 0
    for idx in range(len(real_y)):
        res += -(real_y[idx])*numpy.log(pred_y[idx])-(1-(real_y[idx]))*numpy.log(1-pred_y[idx])
    return res