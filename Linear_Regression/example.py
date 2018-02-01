"""
Linear Regression example.
Use 1 layer linear regression model to calculate add operation.

ex) feature = [3, 5] then output should be 8

If cost graph doesn't converge, then change learning rate more smaller
"""


from Linear_Regression.model import *
from matplotlib.pyplot import *

feature = [[1, 3, 5], [1, 5, 4], [1, 7, 9], [1, 2, 5]] # 4 training examples with 3 features(bias, operand1, operand2)
output = [8, 9, 16, 7]  # 4 training examples output

linear = linear_regression(feature, output, 100, 0.00015) # Create linear regression model with 100 iteration and learning rate as 0.00015
costs = linear.training()                                 # Trainig linear regression model
print linear.forward_propagation([1, 5, 18])              # We can see linear regression model successfully calculate 5 + 18

plot(range(len(costs)), costs)                            # Draw cost graph per # of iterations
show()