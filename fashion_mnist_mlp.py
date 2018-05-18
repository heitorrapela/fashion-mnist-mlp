from __future__ import print_function

# Keras Models
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import fashion_mnist

# Aditional Libs
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# the data, shuffled and split between train and test sets
x_train, y_train, x_valid, y_valid, x_test ,y_test = [], [], [], [], [], []
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000,28,28)
x_test = x_test.reshape(10000,28,28)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255

x_train = x_train.reshape(len(x_train),28,28)
x_test = x_test.reshape(10000,28,28)
x_valid = x_valid.reshape(len(x_valid),28,28)

#print(x_train.shape[0], 'train samples')
#print(x_valid.shape[0], 'valid samples')
#print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(256, activation='tanh', kernel_initializer = 'he_normal' ,input_shape=(28*28,)))
model.add(Dropout(0.4))
model.add(Dense(128, activation='tanh',kernel_initializer = 'he_normal'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='tanh',kernel_initializer = 'he_normal'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='sigmoid',kernel_initializer = 'he_normal'))
optim = keras.optimizers.SGD(lr=0.01, momentum=0.975, decay=2e-06, nesterov=True)

model.compile(loss='categorical_crossentropy',
			  optimizer=optim,
			  metrics=['accuracy'])

history = model.fit(x_train, y_train,
					batch_size=64,
					epochs=100,
					verbose=2,
					validation_data=(x_valid, y_valid))

print(model.summary())
	
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test top 1 accuracy:', score[1])