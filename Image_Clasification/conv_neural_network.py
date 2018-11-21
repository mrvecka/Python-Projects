import DataLoad as load
import tensorflow as tf
import ImageClasificationNeuralNetwork as icnn
import Common as com
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_data_x, train_data_y, test_data_x, test_data_y = load.CreateDataSets("C:\\Users\\Lukas\\Documents\\Python Projects\\TestData\\cifar-10-batches-py\\data_batch_", "rb", 90)

device_name = "/gpu:0"

learning_rate = 0.001
epochs = 10
batch_size = 128

image_placeholder = tf.placeholder(tf.float32, [None,3072])
labels_placeholder = tf.placeholder(tf.float32,[None,2])
is_training = tf.placeholder(tf.bool)

shaped_image_placeholder = tf.reshape(image_placeholder,[-1,32,32,3])


layer1 = icnn.create_new_conv_network(shaped_image_placeholder,3,32,[5,5],[2,2],name='layer1',is_training=is_training)

layer2 = icnn.create_new_conv_network(layer1,32,64,[5,5],[2,2],name='layer2',is_training=is_training)

# layer3 = icnn.create_new_conv_network(layer2,64,64,[5,5],[2,2],name='layer3')

# layer4 = icnn.create_new_conv_network(layer3,64,64,[5,5],[2,2],name='layer4')

# layer5 = icnn.create_new_conv_network(layer4,32,10,[5,5],[2,2],name='layer5')


flattered = tf.reshape(layer2,[-1,8*8*64])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 3072], stddev=0.1), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([3072], stddev=0.1), name='bd1')
dense_layer1 = tf.add(tf.matmul(flattered, wd1), bd1)

dense_layer1 = tf.layers.batch_normalization(dense_layer1,training=is_training)

dense_layer1 = tf.nn.relu(dense_layer1)


# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([3072, 2], stddev=0.1), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([2], stddev=0.1), name='bd2')
dense_layer2 = tf.add(tf.matmul(dense_layer1, wd2), bd2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=labels_placeholder))


# add an optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(labels_placeholder, 1), tf.argmax(dense_layer2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(1000):
            indices = np.random.choice(train_data_x.shape[0], batch_size)            
            image_batch = train_data_x[indices]
            labels_batch = train_data_y[indices]
            
            sess.run(optimizer, 
                            feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: True})

        test_acc = sess.run(accuracy, 
                        feed_dict={image_placeholder: test_data_x, labels_placeholder: test_data_y, is_training: False})
        
        print("Epoch:", (epoch + 1), "test accuracy: {:.5f}".format(test_acc))
            


    # indices = np.random.choice(test_data_x.shape[0], 2000)
    # img_batch = test_data_x[indices]
    # lbl_batch = test_data_y[indices]
    # img_batch = com.Reshape(img_batch,2000,1024)

    print("\nTraining complete!")
    print("Accuracy:",sess.run(accuracy, feed_dict={image_placeholder: test_data_x, labels_placeholder: test_data_y, is_training: False}))


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