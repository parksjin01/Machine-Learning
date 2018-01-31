import numpy

def cost(real_y, pred_y):
    real_y = numpy.matrix(real_y, dtype=numpy.float64)
    pred_y = numpy.matrix(pred_y, dtype=numpy.float64).transpose()

    res = ((pred_y - real_y) * (pred_y - real_y).transpose()).tolist()
    return res[0]