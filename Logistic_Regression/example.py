"""
Logistic Regression example.
Use 1 layer logistic regression model to calculate or operation.

ex) feature = [1, 1] then output should be 1

If cost graph doesn't converge, then change learning rate more smaller
"""


from Logistic_Regression.model import *
from matplotlib.pyplot import *

feature = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]] # 4 training examples with 3 features(bias, operand1, operand2)
output = [0, 1, 1, 1]  # 4 training examples output

logistic = Logistic_regression(feature, output, 100, 0.05)  # Create logistic regression model with 100 iteration and learning rate as 0.05
costs = logistic.training()                                 # Trainig logistic regression model
print logistic.forward_propagation([1, 1, 1])               # We can see logistic regression model successfully calculate 1 || 1

plot(range(len(costs)), costs)                            # Draw cost graph per # of iterations
show()