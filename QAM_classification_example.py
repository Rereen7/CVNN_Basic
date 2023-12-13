# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:13:03 2023

@author: Rayyan Abdalla

    Classification example using complex-valued neural networks implemented by the repository:
        https://github.com/NEGU93/cvnn
        Classification among 6 different categories of complex QAM signals of order (4,16,64,256), BSPK, and random Guassian noise
            dataset comprises 6000 samples for training/Testing, Sample size is 256.
        
        
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models
import cvnn.layers as complex_layers   # Ouw layers!
import scipy.io
from cvnn.losses import ComplexAverageCrossEntropy,ComplexMeanSquareError
from cvnn.layers import complex_input
from cvnn.activations import  mvn_activation



"""
mat_gen_data = scipy.io.loadmat('C:\\Users\\Administrator\\cvnn\\examples\\dataset_random_sampled_10k')

labeled_data = mat_gen_data['random_labeled_data'];


x = labeled_data[:,0:256]
y = labeled_data[:,256]
'''
x = tf.cast(x, tf.complex64)
y = tf.cast(y,tf.complex64)
'''
Train_data = x[0:32000]
Train_labels = y[0:32000]

Test_data = x[32000:40000]
Test_labels = y[32000:40000]



train_data = np.zeros((32000,256,1),dtype='complex64');
test_data = np.zeros((8000,256,1),dtype='complex64');
train_labels = np.zeros((32000,1))
test_labels = np.zeros((8000,1))
"""


mat_gen_data = scipy.io.loadmat('matlab_data\\dataset_random_sampled_qam1k') #'C:\\Users\\Administrator\\cvnn\\examples\\dataset_random_sampled_qam1k'

labeled_data = mat_gen_data['random_labeled_data'];


x = labeled_data[:,0:256]
y = labeled_data[:,256]
'''
x = tf.cast(x, tf.complex64)
y = tf.cast(y,tf.complex64)
'''
Train_data =  x[0:5000]  # x[0:32000]
Train_labels =  y[0:5000] #  y[0:32000]

Test_data =  x[5000:6000]    # x[32000:40000]
Test_labels = y[5000:6000]   # y[32000:40000]



train_data = np.zeros((5000,256,1),dtype='complex64'); # 32000,256,1
test_data = np.zeros((1000,256,1),dtype='complex64');  #8000,256,1
train_labels = np.zeros((5000,1)) #32000,1
test_labels = np.zeros((1000,1)) #8000



for i in range(len(Train_data)):
    for j in range(256):
        train_data[i][j][0] = Train_data[i][j]
    train_labels[i] = Train_labels[i]


for i in range(len(Test_data)):
    for j in range(256):
        test_data[i][j][0] = Test_data[i][j]
    test_labels[i] = Test_labels[i]

train_labels = train_labels.astype(dtype=np.uint)
train_labels=train_labels.reshape([-1])
encoded_train_labels = np.zeros((train_labels.size, train_labels.max()+1), dtype=int)

#replacing 0 with a 1 at the index of the original array
encoded_train_labels[np.arange(train_labels.size),train_labels] = 1 


test_labels = test_labels.astype(dtype=np.uint)
test_labels=test_labels.reshape([-1])
encoded_test_labels = np.zeros((test_labels.size, test_labels.max()+1), dtype=int)

#replacing 0 with a 1 at the index of the original array
encoded_test_labels[np.arange(test_labels.size),test_labels] = 1 


model = models.Sequential()

model.add(complex_input(shape=(256, 1)))
model.add(complex_layers.ComplexConv1D(32,5, activation='cart_relu', input_shape=(256, 1) , dtype=np.complex64))
model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))
model.add(complex_layers.ComplexConv1D(64, 5, activation='cart_relu',dtype=np.complex64))
model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))
# model.add(complex_layers.ComplexDropout(0.5))
# model.add(complex_layers.ComplexConv1D(128, 5, activation='cart_relu',dtype=np.complex64))
# model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))
# model.add(complex_layers.ComplexDropout(0.5))
# model.add(complex_layers.ComplexConv1D(64, 5, activation='cart_relu',dtype=np.complex64))
# model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))



model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(32, activation='cart_relu'))
model.add(complex_layers.ComplexDense(6,activation="softmax_real_with_avg",dtype=np.complex64)) #4
model.summary()

opt =  tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=ComplexAverageCrossEntropy(),
              metrics=['accuracy'])

'''
model = models.Sequential()

model.add(complex_input(shape=(256,1)))
model.add(complex_layers.ComplexFlatten(dtype=np.complex64))
model.add(complex_layers.ComplexDense(128, activation="mvn_activation",dtype=np.complex64))
model.add(complex_layers.ComplexDropout(0.5))
# model.add(complex_layers.ComplexDense(64, activation="mvn_activation",dtype=np.complex64))
# model.add(complex_layers.ComplexDropout(0.5))
# model.add(complex_layers.ComplexDense(128, activation="mvn_activation",dtype=np.complex64))
# model.add(complex_layers.ComplexDropout(0.5))
# model.add(complex_layers.ComplexDense(32,activation="mvn_activation",dtype=np.complex64))
model.add(complex_layers.ComplexDense(4,activation="softmax_real_with_avg",dtype=np.complex64))

model.compile(optimizer='adam',
              loss=ComplexAverageCrossEntropy(),
              metrics=['accuracy'])
model.summary()
'''
history = model.fit(train_data, encoded_train_labels, epochs=10, validation_data=(test_data, encoded_test_labels))
#tf.keras.losses.SparseCategoricalCrossentropy



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

if __name__ == "__main__":
    from importlib import reload
    import os
    import tensorflow
    reload(tensorflow)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'