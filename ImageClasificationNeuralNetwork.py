import tensorflow as tf
import sys

def neural_network_model(data,n_layers,n_classes,n_nodes,device_name = None):
    """ Methode create neural network based on parameters

    @param: data The images data of shape (x,1,y) where y = e.g. 32x32 or 24x24 etc
    @param: n_layers Number of hidden layers except input and output layer
    @param: n_classes The number of classification classes
    @param: n_nodes number of neurons in one layer
    @param: device_name Name of the graphic card which shpould be use for training e.g. /gpu:0


    """
    try:
        if device_name != None:
            with tf.device(device_name):
                return CreateNeuralNetwork(data,n_layers,n_classes,n_nodes)
        else:
            return CreateNeuralNetwork(data,n_layers,n_classes,n_nodes)
    except:
        if n_classes == 0:
            print("Number of classification classes must be at least 2")
            return
        else:
            print("!!! Something went wrong !!!")
            print("Exception:",sys.exc_info()[0])
            return                        

def CreateNeuralNetwork(data,n_layers,n_classes,n_nodes):
    # (input_data * weights) + biases

    hidden_layers = []
    hidden_layers.append({'weights': tf.Variable(tf.random_normal([data.get_shape()[1].value,n_nodes])),
                        'biases': tf.Variable(tf.random_normal([n_nodes]))})

    for i in range(n_layers):
        hidden_layers.append({'weights': tf.Variable(tf.random_normal([n_nodes,n_nodes])),
                                'biases': tf.Variable(tf.random_normal([n_nodes]))})

    hidden_layers.append({'weights': tf.Variable(tf.random_normal([n_nodes,n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))})

    prev_layer = None
    output = None
    for i in range(len(hidden_layers)):
        if i == 0:
            # input
            l1 = tf.add(tf.matmul(data, hidden_layers[i]['weights']), hidden_layers[i]['biases'])
            prev_layer = tf.nn.relu(l1)
        elif i == len(hidden_layers) -1:
            # output
             output = tf.matmul(prev_layer, hidden_layers[i]['weights']) + hidden_layers[i]['biases']
        else:
            l2 = tf.add(tf.matmul(prev_layer, hidden_layers[i]['weights']), hidden_layers[i]['biases'])
            prev_layer = tf.nn.relu(l2)

    return output
