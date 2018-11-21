import tensorflow as tf
import Common as com 
import DataLoad as load

train_data_x, train_data_y, test_data_x, test_data_y = load.CreateDataSets("C:\\Users\\Lukas\\Documents\\Python Projects\\TestData\\cifar-10-batches-py\\data_batch_", "rb", 70)

sess = tf.Session()

saver = tf.train.import_meta_graph("C:\\Users\\Lukas\\Documents\\Python Projects\\Saved models\\my_test_model2.meta")

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
image_paceholder = graph.get_tensor_by_name("image_placeholder:0")
labels_placeholder = graph.get_tensor_by_name("labels_placeholder:0")

restored_op = graph.get_tensor_by_name("accuracy:0")


if len(test_data_x.shape) == 3:
    test_data_x = com.Reshape(test_data_x,test_data_x.shape[0],test_data_x.shape[2])

test_accuracy = sess.run(restored_op, feed_dict = {image_paceholder:test_data_x, labels_placeholder:test_data_y})

print("\n")
print("\n")
print("\n")
print('Test accuracy {:g}'.format(test_accuracy))
print("\n")
print("\n")
print("\n")

sess.close()