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
keep_prob = tf.placeholder(tf.float32)

# Use xavier_init to converge fastly
initializer = tf.contrib.layers.xavier_initializer()

# Create 5 weight variable
W1 = tf.Variable(initializer([NUM_Feature, 6]))
W2 = tf.Variable(initializer([6, 6]))
W3 = tf.Variable(initializer([6, 6]))
W4 = tf.Variable(initializer([6, 6]))
W5 = tf.Variable(initializer([6, NUM_Result]))

# Create 5 Bias variable to converge fastly
B1 = tf.Variable(tf.random_normal([6], stddev = 0.01))
B2 = tf.Variable(tf.random_normal([6], stddev = 0.01))
B3 = tf.Variable(tf.random_normal([6], stddev = 0.01))
B4 = tf.Variable(tf.random_normal([6], stddev = 0.01))
B5 = tf.Variable(tf.random_normal([NUM_Result], stddev = 0.01))

# Calculate each layer's output
hidden1 = tf.nn.relu(tf.matmul(x, W1) + B1)
tf.nn.dropout(hidden1, keep_prob=keep_prob)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + B2)
tf.nn.dropout(hidden2, keep_prob=keep_prob)
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + B3)
tf.nn.dropout(hidden3, keep_prob=keep_prob)
hidden4 = tf.nn.relu(tf.matmul(hidden3, W4) + B4)
tf.nn.dropout(hidden4, keep_prob=keep_prob)
output = tf.matmul(hidden4, W5) + B5

# Cost function and optimizer to learning neural network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Learning 2000 times
for epoch in range(10000):
    _, total_cost = sess.run([optimizer, cost], feed_dict={x: feature, y: result, keep_prob: 0.8})
    if epoch%100 == 0:
        print '-'*100
        print "Epoch:", epoch, 'Cost:', total_cost
        print '-'*100

print "\n\n\n\nTraining is finished, Calculate accuracy"

# Accuracy is approximately 96%
is_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={x: feature,
                                   y: result}))

exit(0)
# For error analyze manually
for idx in range(len(feature)):
    res = sess.run(is_correct, feed_dict={x: feature[idx], y: result[idx]})
    if not res:
        print feature[idx], result[idx], sess.run(tf.argmax(output,1), feed_dict={x: feature[idx], keep_prob: 1})