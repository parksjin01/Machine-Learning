# -*- encoding:utf8 -*-

import tensorflow as tf
import numpy as np
import json

# Read data from data file
# data file form is json
with open('data', 'rb') as f:
    pre_data = json.loads(f.read())
    feature = [map(float, tmp) for tmp in pre_data['feature']]
    result = [map(int, tmp) for tmp in pre_data['output']]

# Calculate number of input/output layer's automatically
NUM_Feature = len(feature[0])
NUM_Result = len(result[0])

# Change 2D list to 2D matrix
feature = np.matrix(feature)
result = np.matrix(result)

# Create tensorflow place holder
x = tf.placeholder(tf.float32, [None, NUM_Feature])
y = tf.placeholder(tf.float32, [None, NUM_Result])

# Create 3 weight variable
W1 = tf.Variable(tf.random_normal([NUM_Feature, 6], stddev = 0.01))
W2 = tf.Variable(tf.random_normal([6, 6], stddev = 0.01))
W3 = tf.Variable(tf.random_normal([6, NUM_Result], stddev = 0.01))

# Normalize feature (feature scaling and mean normalization)
# This is very important, If skip this neural network won't learned properly
norm = tf.nn.l2_normalize(x)

# Calculate each layer's output
hidden1 = tf.nn.softmax(tf.matmul(norm, W1))
hidden2 = tf.nn.softmax(tf.matmul(hidden1, W2))
output = tf.nn.softmax(tf.matmul(hidden2, W3))

# Cost function and optimizer to learning neural network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(0.3).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Learning 2000 times
for epoch in range(2000):
    _, total_cost = sess.run([optimizer, cost], feed_dict={x: feature, y: result})
    if epoch%100 == 0:
        print total_cost


# Accuracy is approximately 80%
is_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={x: feature,
                                   y: result}))