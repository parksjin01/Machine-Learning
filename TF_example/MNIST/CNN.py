import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

iter = 15
initializer = tf.contrib.layers.xavier_initializer()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# Reshape input as 28x28 image
x_reshape = tf.reshape(x, [-1, 28, 28, 1])

# First convolution layer
W1 = tf.Variable(initializer([3, 3, 1, 32]))
L1 = tf.nn.conv2d(x_reshape, W1, strides=[1, 1, 1, 1], padding="SAME")
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Second convolution layer
W2 = tf.Variable(initializer([3, 3, 32, 64]))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

# Artificial neural network to do softmax regression
W3 = tf.Variable(initializer([7 * 7 * 64, 10]))
B3 = tf.Variable(initializer([10]))
output = tf.matmul(L2, W3) + B3

# Cost function and optimizer function to training
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels = y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
for epoch in range(15):
    total_cost = 0

    for _ in range(mnist.train.num_examples/100):
        x_train, y_train = mnist.train.next_batch(100)
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_train, y:y_train})
        total_cost += cost_val
    print epoch+1, total_cost

# Check accuracy
is_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={x: mnist.test.images,
                                   y: mnist.test.labels}))
