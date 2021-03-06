# -*- encoding:utf8 -*-

"""
Goal of this program is create data file from Pre_Data file
Created data file will be used with obesity neural network
"""

import json

# Target designate which data will be extracted from Pre_Data file
Target = ['"나이"', '"104:키"', '"510:몸무게"', '"518:BMI 평가"']

with open("Pre_Data.csv", 'rb') as f:
    data = f.read().split('\n')
    head, body = data[0].split(','), data[1:]
    # Target = head
    tgt_idx = []

    # Find the index of item in target list
    for keyword in Target:
        tgt_idx.append(head.index(keyword))

    # Extract data which is designated in target list from Pre_Data
    feature = []
    output = []
    for line in body:
        tmp = []
        for idx in tgt_idx:
            tmp.append(line.split(',')[idx].strip('"'))
        feature.append(['1']+tmp[:-1])
        output.append(tmp[-1])

# It's not binary classification problem. So doing one-hot encoding
# e.x) 4 -> [0, 0, 0, 0, 1]
#      2 -> [0, 0, 1, 0, 0]
one_hot = {}
for idx in output:
    one_hot[idx] = 1
one_hot = dict(zip(one_hot.keys(), range(len(one_hot.keys()))))
one_hot_output = []
for item in output:
    one_hot_output.append(['0']*len(one_hot.keys()))
    one_hot_output[-1][one_hot[item]] = '1'

# Create help file to analyze one-hot encoding
with open('help', 'w') as f:
    f.write(' '.join(one_hot.keys()))

# Normalize(Feature scaling and mean normalization) data
# Instead of normalizing data with tensorflow, it converge more faster and increase accuracy about 5%
feature = map(list, zip(*[map(float, f) for f in feature]))
mean = [0]
min_max = [1,]
for line in feature[1:]:
    mean.append(sum(line)/len(line))
    min_max.append(max(line)-min(line))

for line in range(len(feature)):
    for value in range(len(feature[0])):
        feature[line][value] = (feature[line][value]-mean[line])/min_max[line]
feature = map(list, zip(*[map(float, f) for f in feature]))

# Changing data to dict form to save data as json format
data = {
    'feature': feature,
    'output': one_hot_output
}

# Save it!
with open('DATA', 'wb') as f:
    f.write(json.dumps(data))
