from Logistic_Regression.cost_function import *
from Logistic_Regression.hypothesis_function import *
from Logistic_Regression.gradient_descent import *
from matplotlib.pyplot import *

feature = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
output = [0, 0, 0, 1]

weight = [[0], [0], [0]]

costs = []

for _ in range(100):
    res = hypotheis(feature, weight)
    # print res
    costs.append(cost(output, res))
    weight = gradient_descent(output, res, feature, weight, 0.1)

print hypotheis(feature, weight)
plot(range(len(costs)), costs)
show()