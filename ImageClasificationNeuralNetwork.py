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


def create_new_conv_network(input_data, num_input_chanels, num_output_channels, filter_shape, pool_shape, name,is_training):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filter_shape = [filter_shape[0],filter_shape[1], num_input_chanels, num_output_channels]

    # initialize weights anddd bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.1),name=name+'_W')

    bias = tf.Variable(tf.truncated_normal([num_output_channels]),name=name+'_b')

    # setup the convolutional layer network
    out_layer = tf.nn.conv2d(input_data,weights,strides=[1,1,1,1],padding='SAME')

    # add bias
    out_layer = tf.add(out_layer,bias)

    # normalization
    out_layer = tf.layers.batch_normalization(out_layer,training=is_training)

    # apply a relu non-linea activation
    out_layer = tf.nn.relu(out_layer)

    # perform max pooling
    ksize = [1, pool_shape[0],pool_shape[1],1]
    strides = [1,2,2,1]
    out_layer = tf.nn.max_pool(out_layer,ksize=ksize,strides= strides,padding='SAME')

    return out_layer