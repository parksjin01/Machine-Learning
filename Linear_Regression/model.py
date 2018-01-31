from Linear_Regression.cost_function import *
from Linear_Regression.hypothesis_function import *
from Linear_Regression.gradient_descent import *
from matplotlib.pyplot import *

feature = [[1, 3, 0], [1, 1, 5], [1, 8, 1], [1, 10, 1]]
output = [3, 6, 9, 11]

weight = [[0], [0], [0]]

costs = []

for _ in range(100):
    res = hypotheis(feature, weight)
    # print res
    # print cost(output, res)
    costs.append(cost(output, res))
    weight = gradient_descent(output, res, feature, weight, 0.01)

print hypotheis([1, 8, 9], weight)
plot(range(len(costs)), costs)
show()