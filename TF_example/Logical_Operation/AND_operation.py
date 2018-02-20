"""
    Logistic Regression model with Tensorflow
    There are no hidden layer, network is composed with only input, output layer
    Doing AND operation
"""

import tensorflow as tf
import numpy as np

# DataSet of AND logical operation
DataSet = {
    'feature':
    # Training example = (Bias, feature1, feature2)
    [
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ],
    'output':
    [
        [0],
        [0],
        [0],
        [1]
    ]
}

# Create tf placeholder for feature and output
x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 1])

# Initialize tf variable for weight between input layer and hidden layer, hidden layer and output layer
W3 = tf.Variable(tf.random_normal([3, 1], stddev = 0.1))

# Result of each layer
L3 = tf.nn.sigmoid(tf.matmul(x, W3))

# Cost function and optimizer to learning model
cost = tf.losses.log_loss(labels=y, predictions=L3)
optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

# Create tensorflow session and init variable
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Learning tensor flow model
for epoch in range(15000):
    _, total_cost = sess.run([optimizer, cost], feed_dict={x:np.matrix(DataSet['feature']), y:np.matrix(DataSet['output'])})
    if epoch%1000 == 0:
        print '-'*100
        print "Epoch: %d, Cost: %lf" %(epoch, total_cost)
        print '-'*100

print "\n\n\n\nLearning finished. Test starts"

is_correct = tf.equal(tf.round(L3), y)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={x: np.matrix(DataSet['feature']),
                                   y: np.matrix(DataSet['output'])}))


