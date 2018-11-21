import tensorflow as tf
import os
print(os.name)
hello = tf.constant('Hello')
sess = tf.Session()
print(sess.run(hello))


        


