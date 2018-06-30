# -*- coding: utf-8 -*-
"""
@CREATETIME: 30/06/2018 14:05 
@AUTHOR: Chans
@VERSION: 
"""
import tensorflow as tf

with tf.variable_scope("foo") :
    v = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
