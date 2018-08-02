# -*- coding: utf-8 -*-
"""
@CREATETIME: 01/07/2018 10:33 
@AUTHOR: Chans
@VERSION: 
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[2]),name='v2')
result = v1 + v2

print(result.name)
init_op = tf.global_variables_initializer()

with tf.Session() as  sess :
    sess.run(init_op)
    graph_def = tf.get_default_graph().as_default()
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
    with tf.gfile.GFile('/Users/Apple/datadata/tf/tf_learning/combined_model.pb','wb') as f:
        f.write(output_graph_def.SerializeToString())

with tf.Session() as sess :
    with gfile.FastGFile('/Users/Apple/datadata/tf/tf_learning/combined_model.pb','rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    resultt = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))


# v = tf.Variable(0,name='v',dtype=tf.float32)
# ema = tf.train.ExponentialMovingAverage(0.99)
# average_op = ema.apply(tf.global_variables())
#
#
# saver = tf.train.Saver(ema.variables_to_restore())
#
# for var in tf.global_variables():
#     print(var)
# with tf.Session() as sess :
#     sess.run(init_op)
#     sess.run(tf.assign(v,10))
#     sess.run(average_op)
#     saver.save(sess,save_path='/Users/Apple/datadata/tf/tf_learning/model.ckpt')
#     print(sess.run([v,ema.average(v)]))
#
#     # saver.save(sess,save_path='/Users/Apple/datadata/tf/tf_learning/model.ckpt')
#     # saver.restore(sess,"/Users/Apple/datadata/tf/tf_learning/model.ckpt")
#     # print(sess.run(result))



# v = tf.Variable(0,name='v',dtype=tf.float32)
# ema = tf.train.ExponentialMovingAverage(0.99)
# saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess :
#     saver.restore(sess,save_path='/Users/Apple/datadata/tf/tf_learning/model.ckpt')
#     print(sess.run(v ))


