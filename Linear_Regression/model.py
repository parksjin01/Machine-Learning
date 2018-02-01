"""
Class of linear regression model
"""


from Linear_Regression.cost_function import *
from Linear_Regression.hypothesis_function import *
from Linear_Regression.gradient_descent import *

class linear_regression():

    """
    Purpose:   Object initialization method
    Parameter: feature, output, # of training iteration, learning rate
    Return:    None
    """
    def __init__(self, x, y, iteration = 100, learning_rate = 0.01):
        self.x = x
        self.y = y
        self.iter = iteration
        self.learning_rate = learning_rate
        self.weight = []

        for _ in xrange(len(x[0])):
            self.weight.append([0])

    """
    Purpose:   Training linear regression model
    Parameter: None
    Return:    List of costs
    """
    def training(self):
        costs = []
        for _ in range(self.iter):                  # Training `iter` times
            res = hypotheis(self.x, self.weight)
            costs.append(cost(self.y, res))
            self.weight = gradient_descent(self.y, res, self.x, self.weight, self.learning_rate)
            print self.weight
        return costs

    """
    Purpose:   Calculate linear regression model's result for given feature
    Parameter: Feature
    Return:    Linear regression model's result for given feature
    """
    def forward_propagation(self, x):
        return hypotheis(x, self.weight)
