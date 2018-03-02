import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

ITER = 15
BATCH = 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
W2 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))

L1 = tf.nn.softmax(tf.matmul(x, W1))
L2 = tf.nn.softmax(tf.matmul(L1, W2))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=L2, labels = y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(ITER):
    total_cost = 0

    for _ in range(mnist.train.num_examples/BATCH):
        x_train, y_train = mnist.train.next_batch(BATCH)
        # print len(y_train[0])
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_train, y:y_train})
        total_cost += cost_val
    print epoch+1, total_cost


is_correct = tf.equal(tf.argmax(L2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={x: mnist.test.images,
                                   y: mnist.test.labels}))