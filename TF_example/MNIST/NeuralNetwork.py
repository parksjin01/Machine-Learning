import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

iter = 15

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

initializer = tf.contrib.layers.xavier_initializer()

W1 = tf.Variable(initializer([784, 256]))
W2 = tf.Variable(initializer([256, 256]))
W3 = tf.Variable(initializer([256, 256]))
W4 = tf.Variable(initializer([256, 256]))
W5 = tf.Variable(initializer([256, 10]))

B1 = tf.Variable(tf.random_normal([256], stddev = 0.01))
B2 = tf.Variable(tf.random_normal([256], stddev = 0.01))
B3 = tf.Variable(tf.random_normal([256], stddev = 0.01))
B4 = tf.Variable(tf.random_normal([256], stddev = 0.01))
B5 = tf.Variable(tf.random_normal([10], stddev = 0.01))

L1 = tf.nn.relu(tf.matmul(x, W1) + B1)
tf.nn.dropout(L1, keep_prob=keep_prob)

L2 = tf.nn.relu(tf.matmul(L1, W2) + B2)
tf.nn.dropout(L2, keep_prob=keep_prob)

L3 = tf.nn.relu(tf.matmul(L2, W3) + B3)
tf.nn.dropout(L3, keep_prob=keep_prob)

L4 = tf.nn.relu(tf.matmul(L3, W4) + B4)
tf.nn.dropout(L4, keep_prob=keep_prob)

output = tf.matmul(L4, W5) + B5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels = y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(15):
    total_cost = 0

    for _ in range(mnist.train.num_examples/100):
        x_train, y_train = mnist.train.next_batch(100)
        # print len(y_train[0])
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_train, y:y_train})
        total_cost += cost_val
    print epoch+1, total_cost


is_correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={x: mnist.test.images,
                                   y: mnist.test.labels}))
