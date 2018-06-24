#
# SimpleMNIST.py
# Simple NN to clasify handwritten digits from NIST dataset
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# we use the tf helper function to pull down  the data from the nist site
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
epochs = 10

# x is placeholder for the 28 x 28 image data
x = tf.placeholder(tf.float32, shape=[None,784])

#y_ is called "y bar" and is a 10 element vector, contaning the predicted probability of each
# digit(0-9) class. Such as [0.14,0.8,0,0,0,0,0,0,0,0.06]
y_ = tf.placeholder(tf.float32,[None,10])

# define wights and balances
w= tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# define our model
y = tf.nn.softmax(tf.matmul(x,w)+b)

#loss is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

#each training step in gradient decent we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialization
init = tf.global_variables_initializer()


sess = tf.Session()

sess.run(init)
for ij in range(epochs):    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})



# evaluate how well the model did. do this by comparing the digits with the highest probability in
# actual (y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,y_:mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()


