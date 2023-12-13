# -*- coding: utf-8 -*-
"""
@author: Rayyan Abdalla
Email: rsabdall@asu.edu

Python implementation of the Matlab Regression example: 
    https://www.mathworks.com/help/deeplearning/ug/train-network-with-complex-valued-data.html
    
    Tensorflow implementation following Complex-valued neural network approach
    Library obtained from repository:
        https://github.com/NEGU93/cvnn
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models
import cvnn.layers as complex_layers   # Ouw layers!
import scipy.io
from cvnn.losses import ComplexAverageCrossEntropy,ComplexMeanSquareError
from cvnn.layers import complex_input

matlab_workspace = scipy.io.loadmat('matlab_data/matlab_data')
'''
matlab_data = matlab_workspace['data'];

arr = np.zeros((500,76,2),dtype='complex64')
for i in range(len(matlab_data)):
    for j in range(2):
        arr[i,:,j] = matlab_data[i][0][j]
        
'''
Train_data = matlab_workspace['XTrain'];
Test_data = matlab_workspace['XTest'];

train_arr = np.zeros((400,76,2),dtype='complex64');

for i in range(len(Train_data)):
    for j in range(2):
        train_arr[i,:,j] = Train_data[i][0][j]

test_arr = np.zeros((100,76,2),dtype='complex64');

for i in range(len(Test_data)):
    for j in range(2):
        test_arr[i,:,j] = Test_data[i][0][j]
        


train_freq = matlab_workspace['TTrain'] 
test_freq = matlab_workspace['TTest']         

train_labels = matlab_workspace['Train_nlabels']
test_labels = matlab_workspace['Test_nlabels']

# test_labels.astype(np.complex64)
# train_labels.astype(np.complex64)
#arr = np.reshape(arr,(500,76,2))


#===================================================================================
"""
model = models.Sequential()
model.add(complex_layers.ComplexConv1D(32, 5, activation='cart_relu', input_shape=(76, 2) , dtype=np.complex64))
model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))
model.add(complex_layers.ComplexConv1D(64, 5, activation='cart_relu',dtype=np.complex64))
model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))



model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(64, activation='cart_relu',dtype=np.float32))
model.add(complex_layers.ComplexDense(4,dtype=np.float32))
model.summary()
"""

# train_freq = (train_freq-np.mean(train_freq))/np.std(train_freq)
# test_freq = (test_freq-np.mean(train_freq))/np.std(train_freq)


model = models.Sequential()

model.add(complex_input(shape=(76, 2)))
model.add(complex_layers.ComplexConv1D(16, 5, activation='cart_relu', input_shape=(76, 2) , dtype=np.complex64))
model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))
model.add(complex_layers.ComplexConv1D(32, 5, activation='cart_relu',dtype=np.complex64))
model.add(complex_layers.ComplexAvgPooling1D(dtype=np.complex64))


model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(16, activation='cart_relu'))
model.add(complex_layers.ComplexDense(1,activation="cart_relu"))
model.summary()

opt =  tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=ComplexMeanSquareError())


history = model.fit(train_arr, train_freq, epochs=30, validation_data=(test_arr, test_freq))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()



plt.figure()
pred = model.predict(test_arr)
plt.plot(test_freq-np.mean(test_freq),label='target frequency')
plt.plot(np.abs(pred)-np.mean(np.abs(pred)),label='Predicted frequency')
plt.legend(loc='lower right')
plt.ylim([-4, 4])
plt.xlabel('Test sequences')
plt.ylabel('Frequency')
plt.show()
