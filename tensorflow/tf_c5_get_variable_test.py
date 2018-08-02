# -*- coding: utf-8 -*-
"""
@CREATETIME: 30/06/2018 14:05 
@AUTHOR: Chans
@VERSION: 
"""
import tensorflow as tf




with tf.variable_scope("foo" ) :
    v = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))


with tf.variable_scope("foo" ,reuse= True) :
    v1 = tf.get_variable("v",shape=[1])  # directly gains values of 'V'
    print(v1==v)

with tf.variable_scope("root") :
    print(tf.get_variable_scope().reuse)

    with tf.variable_scope("foo",reuse=True) :
        print(tf.get_variable_scope().reuse)
        with tf.variable_scope("bar") :
            print(tf.get_variable_scope().reuse)
    print(tf.get_variable_scope().reuse)

v1 = tf.get_variable("v" ,shape=[1])
print(v1.name)

with tf.variable_scope("foo" ,reuse=True):
    v2 = tf.get_variable("v",[1])
    print(v2.name)

with tf.variable_scope("xx" ) :
    with tf.variable_scope("bar" ) :
        v3 = tf.get_variable("v" , [1])
        print(v3.name)
    v4 = tf.get_variable('v',[1])
    print(v4.name)