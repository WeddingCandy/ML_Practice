import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# graph = tf.Graph()
# with graph.as_default():
#     foo = tf.Variable(3, name='foo')
#     bar = tf.Variable(2, name='bar')
#     result = foo + bar
#     initialize = tf.global_variables_initializer()
#
# print(result)  # Tensor("add:0", shape=(), dtype=int32)
# with tf.Session(graph=graph) as sess:
#     sess.run(initialize)
#     res = sess.run(result)
# print(res)  # 5
#
#
# weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
# biases = tf.Variable(tf.zeros([200]), name="biases")
#
# Init_ab = tf.variables_initializer([a,b],name="init_ab")
#
# print(tf.get_default_graph().as_graph_def())
# const = tf.constant(1.0,name="constant")

mnist = input_data.read_data_sets("/Users/Apple/PycharmProjects/learn_ml/tensorflow/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()



with tf.Session() as sess:
    start = time.time()
    sess.run(init)
    for i in range(50000):
        batch_xs, batch_ys = mnist.train.next_batch(500)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    end = time.time() - start