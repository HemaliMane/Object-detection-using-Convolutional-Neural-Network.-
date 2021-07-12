from __future__ import division, print_function, absolute_import
# library for optmising inference
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow as tf
# Higher level API tflearn
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np

# Data loading and preprocessing
#helper functions to download the CIFAR 10 data and load them dynamically

#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tflearn
import tensorflow as tf
from PIL import Image
#%matplotlib inline
#for writing text files
import glob
import os     
import random 
#reading images from a text file
from tflearn.data_utils import image_preloader
import math
from tensorflow.python.tools import freeze_graph

IMAGE_FOLDER = 'D:/cnn/dataset'
TRAIN_DATA = 'D:/cnn/training_data.txt'
TEST_DATA = 'D:/cnn/test_data.txt'
VALIDATION_DATA = 'D:/cnn/validation_data.txt'
train_proportion=0.8
test_proportion=0.2
#validation_proportion=0.1

#read the image directories
filenames_image = os.listdir(IMAGE_FOLDER)
#shuffling the data is important otherwise the model will be fed with a single class data for a long time and 
#network will not learn properly
random.shuffle(filenames_image)

#total number of images
total=len(filenames_image)
##  *****training data******** 
fr = open(TRAIN_DATA, 'w')
train_files=filenames_image[0: int(train_proportion*total)]
for filename in train_files:
    if filename[0:5] == 'Apple':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
    elif filename[0:6] == 'Banana':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
    elif filename[0:5] == 'Guava':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 2\n')
    elif filename[0:4] == 'Kiwi':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 3\n')
    elif filename[0:5] == 'Lemon':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 4\n')
    elif filename[0:6] == 'Lychee':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 5\n')
    elif filename[0:6] == 'Orange':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 6\n')
    elif filename[0:4] == 'Pear':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 7\n')
    elif filename[0:11] == 'Pomegranate':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 8\n')
    elif filename[0:10] == 'Strawberry':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 9\n')

fr.close()
##  *****testing data******** 
fr = open(TEST_DATA, 'w')
test_files=filenames_image[int(math.ceil(train_proportion*total)):int(math.ceil((train_proportion+test_proportion)*total))]
for filename in test_files:
    if filename[0:5] == 'Apple':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
    elif filename[0:6] == 'Banana':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
    elif filename[0:5] == 'Guava':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 2\n')
    elif filename[0:4] == 'Kiwi':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 3\n')
    elif filename[0:5] == 'Lemon':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 4\n')
    elif filename[0:6] == 'Lychee':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 5\n')
    elif filename[0:6] == 'Orange':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 6\n')
    elif filename[0:4] == 'Pear':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 7\n')
    elif filename[0:11] == 'Pomegranate':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 8\n')
    elif filename[0:10] == 'Strawberry':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 9\n')

fr.close()

##  *****validation data******** 
fr = open(VALIDATION_DATA, 'w')
valid_files=filenames_image[int(math.ceil((train_proportion+test_proportion)*total)):total]
for filename in valid_files:
    if filename[0:5] == 'Apple':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 0\n')
    elif filename[0:6] == 'Banana':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 1\n')
    elif filename[0:5] == 'Guava':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 2\n')
    elif filename[0:4] == 'Kiwi':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 3\n')
    elif filename[0:5] == 'Lemon':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 4\n')
    elif filename[0:6] == 'Lychee':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 5\n')
    elif filename[0:6] == 'Orange':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 6\n')
    elif filename[0:4] == 'Pear':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 7\n')
    elif filename[0:11] == 'Pomegranate':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 8\n')
    elif filename[0:10] == 'Strawberry':
        fr.write(IMAGE_FOLDER + '/'+ filename + ' 9\n')

fr.close()



#Importing data
X_train, Y_train = image_preloader(TRAIN_DATA, image_shape=(64,64),mode='file', categorical_labels=True,normalize=True)
X_test, Y_test = image_preloader(TEST_DATA, image_shape=(64,64),mode='file', categorical_labels=True,normalize=True)
#X_val, Y_val = image_preloader(VALIDATION_DATA, image_shape=(56,56),mode='file', categorical_labels=True,normalize=True)



print ("Dataset")
print ("Number of training images {}".format(len(X_train)))
print ("Number of testing images {}".format(len(X_test)))
#print ("Number of validation images {}".format(len(X_val)))
print ("Shape of an image {}" .format(X_train[1].shape))
print ("Shape of label:{} ,number of classes: {}".format(Y_train[1].shape,len(Y_train[1])))


#input image
x=tf.placeholder(tf.float32,shape=[None, 64, 64, 3] , name="ipnode")
#input class
y_=tf.placeholder(tf.float32,shape=[None, 10] , name='input_class')

plt.imshow(X_train[1])
plt.axis('off')
plt.title('Sample image with label {}'.format(Y_train[1]))
plt.show()

# AlexNet architecture

input_layer=x
network = conv_2d(input_layer,32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 10, activation='linear')
y_predicted=tf.nn.softmax(network , name="opnode")
'''

input_layer=x
network = conv_2d(input_layer,96, 11, activation='relu')
network = max_pool_2d(network, 3)
network = conv_2d(network, 256, 5, activation='relu')
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3)
network = fully_connected(network, 4096, activation='relu')
network = fully_connected(network, 10, activation='linear')
y_predicted=tf.nn.softmax(network , name="opnode")
'''

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted+np.exp(-10)), reduction_indices=[1]))
#optimiser -
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#TensorFlow session
sess = tf.Session()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#tensorboard for better visualisation
writer =tf.summary.FileWriter('tensorboard/', sess.graph)
epoch=30 # run for more iterations according your hardware's power
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size=32
no_itr_per_epoch=len(X_train)//batch_size
n_test=len(X_test) #number of test samples


# Commencing training process
for iteration in range(epoch):
    print("Iteration no: {} ".format(iteration))

    previous_batch=0
    # Do our mini batches:
    for i in range(no_itr_per_epoch):
        current_batch=previous_batch+batch_size
        x_input=X_train[previous_batch:current_batch]
        x_images=np.reshape(x_input,[batch_size,64,64,3])
        print(current_batch)

        y_input=Y_train[previous_batch:current_batch]
        y_label=np.reshape(y_input,[batch_size,10])
        previous_batch=previous_batch+batch_size

        _,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_images,y_: y_label})
        #if i % 100==0 :
            #print ("Training loss : {}" .format(loss))



    x_test_images=np.reshape(X_test[0:n_test],[n_test,64,64,3])
    y_test_labels=np.reshape(Y_test[0:n_test],[n_test,10])
    Accuracy_test=sess.run(accuracy,
                           feed_dict={
                        x: x_test_images ,
                        y_: y_test_labels
                      })
    # Accuracy of the test set
    Accuracy_test=round(Accuracy_test*100,2)
    print("Accuracy ::  Test_set {} %  " .format(Accuracy_test))





saver = tf.train.Saver()
model_directory='D:/cnn/model3'
#saving the graph
tf.train.write_graph(sess.graph_def, model_directory, 'savegraph.pbtxt')

saver.save(sess, 'D:/cnn/model3/model.ckpt')
# Freeze the graph
MODEL_NAME = 'cnn'
input_graph_path = 'D:/cnn/model3/savegraph.pbtxt'
checkpoint_path = 'D:/cnn/model3/model.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "opnode"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'D:/cnn/model3/frozen_model_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'D:/cnn/model3/optimized_inference_model_'+MODEL_NAME+'.pb'
clear_devices = True
#Freezing the graph and generating protobuf files
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
#Optimising model for inference only purpose
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        sess.graph_def,
        ["ipnode"], # an array of the input node(s)
        ["opnode"], # an array of output nodes
        tf.float32.as_datatype_enum)

with tf.gfile.GFile(output_optimized_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())
sess.close()
