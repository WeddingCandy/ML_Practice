# encoding = utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

input_node = 784
out_node = 10
layer1_node = 1000
batch_size = 100
learning_rate_base = 0.5
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_steps = 5000
moving_average_decay = 0.99

## compute the result of FNN
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) )+  biases1
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return  tf.matmul(layer1,avg_class.average(weights2) )+ avg_class.average(biases2)

def inference2(input_tensor,reuse=False):
    with tf.variable_scope('layer1' ,reuse= reuse) :
        weights = tf.get_variable("weights",[input_node,layer1_node],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases" , [layer1_node] , initializer= tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    with tf.variable_scope('layer2' ,reuse= reuse) :
        weights = tf.get_variable("weights",[input_node,layer1_node],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases" , [layer1_node] , initializer= tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1,weights) + biases)
        return  layer2




## model trainning process
def train(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_node], name='x-input')
    y_ = inference2(x)
    # generate hidden layer paras
    weighs1 = tf.Variable(tf.truncated_normal(shape=[input_node,layer1_node] ,stddev=0.1,name='weights1'))
    biases1 = tf.Variable(tf.constant(0.1,shape=[layer1_node]))
    # generate output layer paras
    weighs2 = tf.Variable(tf.truncated_normal(shape=[layer1_node,out_node] ,stddev=0.1,name='weights2'))
    biases2 = tf.Variable(tf.constant(0.1,shape=[out_node]))
    y = inference(x,None,weighs1,biases1,weighs2,biases2)



    # identify train loopers and Exponential Moving Average concerned
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,weighs1,biases1,weighs2,biases2)


    # compute cross entropy and its mean
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y ,
                                                                   labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # compute loss function
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regularizer(weighs1) + regularizer(weighs2)
    loss = cross_entropy_mean + regularization

    # set exponential decay learning rate
    learning_rate = tf.train.exponential_decay(learning_rate_base,
                                               global_step,
                                               mnist.train.num_examples / batch_size,
                                               learning_rate_decay,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # BPNN updates paras and updates moving average of each para
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    # compute accuracy rate
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initial op and start train
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images , y_:mnist.test.labels}
        # 循环的训练神经网络。
        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is %g "
                      " ,test accuracy using average model is %g"% (i, validate_acc,test_acc))

            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (training_steps, test_acc)))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()