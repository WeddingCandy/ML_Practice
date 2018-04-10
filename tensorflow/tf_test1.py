import tensorflow as tf

a = tf.constant([1,2],name='a')
b = tf.constant([2,3],name='b')

result = a+b

tf.get_default_graph()

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v",initializer=tf.zeros_initializer(),shape=[1])

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))


g = tf.Graph()
with g.device('/gpu:0'):
    result2 = a+b