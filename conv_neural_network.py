import DataLoad as load
import tensorflow as tf
import ImageClasificationNeuralNetwork as icnn
import Common as com
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_data_x, train_data_y, test_data_x, test_data_y = load.CreateDataSets("C:\\Users\\Lukas\\Documents\\Python Projects\\TestData\\cifar-10-batches-py\\data_batch_", "rb", 70)


learning_rate = 0.0001
epochs = 10
batch_size = 100

image_placeholder = tf.placeholder(tf.float32, [None,3072])
labels_placeholder = tf.placeholder(tf.float32,[None,2])

shaped_image_placeholder = tf.reshape(image_placeholder,[-1,32,32,1])


layer1 = icnn.create_new_conv_network(shaped_image_placeholder,1,32,[5,5],[2,2],name='layer1')

layer2 = icnn.create_new_conv_network(layer1,32,64,[5,5],[2,2],name='layer2')


flattered = tf.reshape(layer2,[-1,8*8*64])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.add(tf.matmul(flattered, wd1), bd1)
dense_layer1 = tf.nn.relu(dense_layer1)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 2], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([2], stddev=0.01), name='bd2')
dense_layer2 = tf.add(tf.matmul(dense_layer1, wd2), bd2)
y_ = tf.nn.softmax(dense_layer2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=labels_placeholder))


# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(labels_placeholder, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(1000):
            indices = np.random.choice(train_data_x.shape[0], batch_size)
            image_batch = train_data_x[indices]
            labels_batch = train_data_y[indices]
            image_batch = com.Reshape(image_batch,batch_size,3072)
            
            _, c = sess.run([optimiser, cross_entropy], 
                            feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch})
            avg_cost = c

        if len(test_data_x.shape) == 3:
            test_data_x = com.Reshape(test_data_x,test_data_x.shape[0],test_data_x.shape[2])

        test_acc = sess.run(accuracy, 
                       feed_dict={image_placeholder: test_data_x, labels_placeholder: test_data_y})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={image_placeholder: test_data_x, labels_placeholder: test_data_y}))


# with tf.Session() as sess:
#     # initialise the variables
#     sess.run(init_op)
#     total_batch = int(len(mnist.train.labels) / batch_size)
#     for epoch in range(epochs):
#         avg_cost = 0
#         for i in range(total_batch):
#             batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
#             _, c = sess.run([optimiser, cross_entropy], 
#                             feed_dict={image_placeholder: batch_x, labels_placeholder: batch_y})
#             avg_cost += c / total_batch
#         test_acc = sess.run(accuracy, 
#                        feed_dict={image_placeholder: mnist.test.images, labels_placeholder: mnist.test.labels})
#         print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

#     print("\nTraining complete!")
#     print(sess.run(accuracy, feed_dict={image_placeholder: mnist.test.images, labels_placeholder: mnist.test.labels}))