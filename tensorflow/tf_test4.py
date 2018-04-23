# -*- coding: utf-8 -*-
#!/usr/bin/python
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v",initializer=tf.zeros_initializer()(shape=[1]))
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[1]))
    a = tf.constant([1, 2], name='a', dtype=tf.int32)
    b = tf.constant([3, 4], name='b', dtype=tf.int32)
    result = a + b
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1,dtype=tf.float32))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1,dtype=tf.float32))
    x = tf.placeholder(shape=[3,2],name="input",dtype=tf.float32)

    hidden_layer = tf.matmul(x,w1)
    y = tf.matmul(hidden_layer,w2)
    cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    learning_rate = 0.001
    train_step =tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)



# with tf.Session(graph=g1) as sess:
#      init = tf.global_variables_initializer()
#      sess.run(init)
#      with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))



# with tf.Session(graph=g2) as sess:
#     init =tf.global_variables_initializer()
#     sess.run(init)
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))
#         print(sess.run(result))
#         print(sess.run(train_step,feed_dict={x:[[0.7,0.9],[0.1,0.4,],[0.5,0.8]]}))


## Exercice for BPNN
g3 = tf.Graph()
with g3.as_default():
    batch_size = 8
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1,dtype=tf.float32))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1,dtype=tf.float32))

    x =  tf.placeholder(shape=(None,2),name='x-input',dtype=tf.float32)
    y_ = tf.placeholder(shape=(None,1),name='y-input',dtype=tf.float32)

    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    learning_rate = 0.001
    train_step =tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    rdm = np.random.RandomState(1)
    data_size =128
    X = rdm.rand(data_size,2)
    Y = [[int(x1+x2 < 1)] for (x1,x2) in X ]


with tf.Session(graph=g3) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % data_size
        end = min(start + batch_size ,data_size)
        # print(start,end)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d trainning step(s),cross entropy on all data is %g" % (i,total_cross_entropy))
