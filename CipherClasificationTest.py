import DataLoad as load
import tensorflow as tf
import numpy as np
import Common as com
import ImageClasificationNeuralNetwork as my_nn
import cv2

train_data_x, train_data_y, test_data_x, test_data_y = load.CreateDataSets("C:\\Users\\Lukas\\Documents\\Python Projects\\TestData\\cifar-10-batches-py\\data_batch_", "rb", 70)
#print (train_data_x[0])
#load.LoadCustomImages("C:\\Users\\Lukas\\Pictures", "C:\\Users\\Lukas\\Pictures\\labels.txt")
#print(loadedFiles)

n_nodes = 1000
n_classes = 2
epochs = 10
device_name = "/gpu:0"


batch_size = 100
learning_rate = 0.005
max_step = 1000


with tf.device(device_name): 

    image_placeholder = tf.placeholder(tf.float32, shape = [None, 3072], name="image_placeholder")
    labels_placeholder = tf.placeholder(tf.int64, shape = [None],name="labels_placeholder")


    weights = tf.Variable(tf.zeros([3072, n_classes]),name="wights")
    biases = tf.Variable(tf.zeros([n_classes]),name="biases")

    # trenovanie

    # computation model
    #model = tf.nn.softmax(tf.matmul(image_paceholder, weights) + biases)

    prediction = my_nn.neural_network_model(image_placeholder,7,n_classes,n_nodes,device_name)
    # Define loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = labels_placeholder))
    # Define training operation
    train_step = tf.train.AdamOptimizer().minimize(loss)
    # koniec trenovania


    # testovanie 
    
    # get prediction for image from video
    maxs = tf.argmax(prediction, 1)
    acc = tf.cast(maxs, tf.float32)
    
    # Operation comparing prediction with true label
    correct_prediction = tf.equal(maxs, labels_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

    # koniec tstovania
    init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()

sess.run(init)
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(max_step):
        indices = np.random.choice(train_data_x.shape[0], batch_size)
        image_batch = train_data_x[indices]
        labels_batch = train_data_y[indices]
        image_batch = com.Reshape(image_batch,batch_size,3072)

        train_accuracy = sess.run(train_step, feed_dict =  {image_placeholder:image_batch, labels_placeholder:labels_batch})
        #print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
        #epoch_loss +=train_accuracy[1]
        #print('Epoch loss {:g}'.format(train_accuracy))

# if len(test_data_x.shape) == 3:
#     test_data_x = com.Reshape(test_data_x,test_data_x.shape[0],test_data_x.shape[2])

# test_accuracy = sess.run(accuracy, feed_dict =  {image_placeholder:test_data_x, labels_placeholder:test_data_y})

cap = cv2.VideoCapture('C:\\Users\\Lukas\\Documents\\Python Projects\\TestData\\cats.avi')
i = 0
while(cap.isOpened()):
    if i % 10 == 0:        
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = load.GetImageAsArray(frame)
        result = com.Reshape(result,1,result.shape[0])
        res = sess.run(acc, feed_dict = {image_placeholder:result, labels_placeholder:[1]})
        
        print("Prediction for image is {:g}".format(res[0]))
    i+=1


cap.release()
cv2.destroyAllWindows()

# print("\n")
# print("\n")
# print("\n")
# print('Test accuracy {:g}'.format(test_accuracy))
# print("\n")
# print("\n")
# print("\n")

saver.save(sess,"C:\\Users\\Lukas\\Documents\\Python Projects\\Saved models\\my_test_model2")
sess.close()
