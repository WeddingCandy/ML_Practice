# encoding = utf-8

import tensorflow as tf
a = tf.constant([1.0,2.0], name='a')
b = tf.constant([2.0,3.0], name='b')


g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v',shape=[1] , initializer=tf.zeros_initializer())

with tf.Session(graph=g1) as sess :
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable('v')))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('v' , shape=[2,2] ,initializer=tf.ones_initializer())

with tf.Session(graph= g2) as sess :
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable('v')))


# g = tf.Graph()
# with g.device('/gpu:0'):
#     result = a+b


weights = tf.Variable(tf.random_normal([2,3],mean=0,stddev=2))
biases = tf.Variable(tf.zeros([3]))
w2 = tf.Variable(weights.initialized_value())
w3 = tf.Variable(weights.initialized_value() * 2.0)

print(w2,w3)

