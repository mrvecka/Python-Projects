#
# house price prediction.py
#
# this is very simple prediction of house prices based on house size, implemented in tensorflow. This code is part of Pluralsight's coourse
#
# 

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generate house sizes between 1000 and 3500 square meters
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high = 3500, size= num_house)

# generate house prices from house size with a random nooise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low = 20000,high = 70000,size = num_house)
 
#plt.plot(house_size,house_price,"bx") # bx = blue x
#plt.ylabel("Price")
#plt.xlabel("Size")
#plt.show()


#you need to normalize values to prevent under/overflow

def normalize(array):
    return (array - array.mean()) / array.std()


# define number of training samles 0,7 = 70%. we can take first since the valueas are randomized
num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Set up Tensprflow placeholders that get updated as we descend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float",name="price")

# Define the variables holding the size_factor and price_offset we set during training
# We initialize them to some random values based on the normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

# Define the operation for the prediction values - predicted price = (size_factor * house_size) + price_offset
# Notice the use of the tensorflow add and multiply function. These add the operations to the computation graph
# AND the tensorflow methods understand how to deal with Tensors. Therefore do not try to use numpy or other library
# methodes
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

# Define the loss Function (how much error) - mean square error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price,2)) / (2* num_train_samples)

# optimizer learning rate. The size of the steps down the gradient
learning_rate = 0.1

# Define a gradient descent optimizer that will minimize the loss defined in the opration
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# At this point the tensorflow variables are uninitialized
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    num_training_iter = 50

    #keep iterating the training data
    for iteration in range(num_training_iter):
        
        #fit all training data
        for(x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size:x,tf_price: y})

        # display current status
        if (iteration +1) % display_every == 0:
            c= sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm,tf_price:train_price_norm})
            print("iteration #:", '%04d' % (iteration +1), "cost=","{:.9f}".format(c),\
            "size_factor=",sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))


    print("Optimalization finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm, tf_price:train_price_norm})
    print("Trained cost=", training_cost, "size_factor=",sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset),'\n')





