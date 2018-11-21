import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# HELPERS

# WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

# BIASES
def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

# CONV2D FUNCTION
def conv2d(x,W):
    # x --> [batch_size,Height,Width,Channels] => Channels = 1 | 3 (r,g,b) | n
    # W --> actual kernel [filter Height, filter Width, Channels IN, Channels OUT]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# POOLING
def max_pool_2by2(x):
    # x --> [batch_size,Height,Width,Channels] => Channels = 1 | 3 (r,g,b) | n
    # ksize => kernel size 2x2 and one dimension for grayskale image
    # strides => how fast we gonna move kernel in each dimension
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONAL LAYER
def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W) + b)

# FULLY CONNECTED LAYER
def fully_connected(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,W) + b

# PLACEHOLDERS 
x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])

# LAYERS
x_image = tf.reshape(x,[-1,28,28,1])

# compute 32 features for every 5x5 filter
conv_1 = convolutional_layer(x_image,shape=[5,5,1,32]) # 1 for grayskale 32 for features to compute
conv_1_pooling = max_pool_2by2(conv_1)

conv_2 = convolutional_layer(conv_1_pooling,shape=[5,5,32,64])
conv_2_pooling = max_pool_2by2(conv_2)

conv_2_flat = tf.reshape(conv_2_pooling,[-1,7*7*64])
fully_connected_layer_one = tf.nn.relu(fully_connected(conv_2_flat,1024)) # 1024 is number of neurons

# DROPOUT
hold_prob = tf.placeholder(tf.float32)
full_dropout = tf.nn.dropout(fully_connected_layer_one,keep_prob=hold_prob)

y_predicted = fully_connected(full_dropout,10) # 10 classes

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predicted))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 5000
batch_size = 128
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x:batch_x,y:batch_y,hold_prob:0.5})

        if i%100 == 0:
            print("ON STEP: {}".format(i))
            print("ACCURACY: ")

            matches = tf.equal(tf.argmax(y_predicted,1),tf.argmax(y,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc, feed_dict={x:mnist.test.images,y:mnist.test.labels,hold_prob:1.0}))
            print('\n')
