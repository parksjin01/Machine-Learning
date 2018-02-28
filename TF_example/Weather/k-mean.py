# -*- encoding:utf8 -*-

import csv
import math
import random
import matplotlib.pyplot as plt
import Pre_processing
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
Read csv data (Korean people's height and weight)
Form Height,Weight
"""
# with open('DATA.csv', 'rb') as f:
#     data = csv.reader(f)
#     height = []
#     weight = []
#     for tmp in data:
#         height.append(float(tmp[0]))
#         weight.append(float(tmp[1]))

height = []
weight = []
data = Pre_processing.Preprocess('./data/day', [u'평균기온', u'평균상대습도'])

for idx in range(len(data.feature)):
    height.append(data.feature[idx][0])
    weight.append(data.feature[idx][1])


# Create cluster class
class cluster():

    # Constructor
    # X: weight, Y: height, name: index of cluster instance in list
    # Data: Dict of data(x,y) which is close to cluster instance
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name
        self.data = {
            'x':[],
            'y':[]
        }

    # Add new close data
    def append(self, x, y):
        self.data['x'].append(x)
        self.data['y'].append(y)

    # Calculate distance between data and cluster
    def distance(self, x, y):
        return math.sqrt(math.pow(self.x-x, 2) + math.pow(self.y-y, 2)), self.name

    # Move cluster position to group's middle point
    def move(self):
        self.x = sum(self.data['x'])/len(self.data['x'])
        self.y = sum(self.data['y']) / len(self.data['y'])

    # Clear dict
    def clear(self):
        self.data = {
            'x': [],
            'y': []
        }

# Number of cluster centroid
NUM_CLUSTER = 4

# Initialize cluster centroid
init_centroid = random.sample(range(len(height)), NUM_CLUSTER)
centroid = []
for idx, value in enumerate(init_centroid):
    centroid.append(cluster(weight[value], height[value], idx))
prev_centroid = {
    'x':[],
    'y':[]
}

# Learning 100 times and move cluster centroid 100 times to find proper position
for _ in range(100):
    for cluster_idx in range(NUM_CLUSTER):
        centroid[cluster_idx].clear()
    print 1

    # Find closest cluster centroid for each data
    for idx in range(len(weight)):
        tmp = []
        for cluster_idx in range(NUM_CLUSTER):
            tmp.append(centroid[cluster_idx].distance(weight[idx], height[idx]))
        cluster_idx = min(tmp)[1]
        centroid[cluster_idx].append(weight[idx], height[idx])

    # Calculate group's middle point and move cluster centroid to that point
    for idx in range(NUM_CLUSTER):
        centroid[idx].move()

# Saving cluster centroid coordinate information in list to mapping data in 2D
for idx in range(NUM_CLUSTER):
    prev_centroid['x'].append(centroid[idx].x)
    prev_centroid['y'].append(centroid[idx].y)

# Drawing scatter graph to present data and cluster centroid in 2D
# Blue dot: data, Red dot: cluster centroid

# plt.scatter(weight, height)

colors = ["blue", "green", "yellow", "pink"]
for idx in range(NUM_CLUSTER):
    # for x, y in centroid[idx]:
    plt.scatter(centroid[idx].data['x'], centroid[idx].data['y'], color=colors[idx])

# plt.scatter([x for x, y in centroid[0]],[y for x, y in centroid[0]], color='blue')
# plt.scatter([x for x, y in centroid[1]],[y for x, y in centroid[1]], color='green')
# plt.scatter([x for x, y in centroid[2]],[y for x, y in centroid[2]], color='yellow')
# plt.scatter([x for x, y in centroid[3]],[y for x, y in centroid[3]], color='pink')
plt.scatter(prev_centroid['x'], prev_centroid['y'], color='red')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()