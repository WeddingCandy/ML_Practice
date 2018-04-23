import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


mnist = input_data.read_data_sets("/Users/Apple/PycharmProjects/learn_ml/tensorflow/MNIST_data/", one_hot=True)
# Parameters
learning_rate = 0.015
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

"""
import tensorflow as tf

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
#
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
#
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
#
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
#
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#
# 结果返回一个Tensor，这个输出，就是我们常说的feature map

oplist=[]
# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([1 ,1 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 2"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 3"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 4"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 5"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 6"])


# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,2,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 7"])


# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([4, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,2,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 8"])

with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer())
    for aop in oplist:
        print("----------{}---------".format(aop[1]))
        print(a_sess.run(aop[0]))
        print('---------------------\n\n')
"""



def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

# Store layers weight & bias

if __name__ == "__main__":
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), # 5x5 conv, 1 input, 32 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
        'out': tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))



