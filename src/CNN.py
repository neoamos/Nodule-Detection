
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
import random
import os
from Data import DataHandler
handler = DataHandler()
K.set_image_dim_ordering('th')



num_classes = 2



# Create the model
model = Sequential()

#Runs 64 (3,5,5) Kernels (features) over the image
#Run activation maps through relu to return 0 for negative values
#Constrains the Kernel to be size 3 at most?
#The output now has 32 activation maps, each for a different feature
model.add(Conv3D(64, (3, 5, 5), input_shape=(1, 10, 16, 16), padding='valid', activation='relu'))
#Sets .2 of values to 0 to avoid overfitting
model.add(Dropout(0.2))
#Runs 64 (2,2,2) Kernels (larger features) over the 64 activation maps that we have so far. Each kernel has a different value for
#Each activation map, and per kernel the activation map values are summed
#Output is ran through relu to have values of 0 when negative
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='valid'))
model.add(Conv3D(64, (1, 3, 3), activation='relu', padding='valid'))
#Flattens out the activation maps into a 1 dimensional array of values
model.add(Flatten()) #You flatten so that you can eventually get 1 array, you do not want a ton of activation maps
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



#Want to minimize log softmax loss after this.
#Softmax activation is used to give

# Compile model
epochs = 25 # Runs over whole dataset 25 times
lrate = 0.01 #Trains weights with a learning rate of 0.01
decay = lrate/epochs #Decays the influence of the weights over time as you have more epochs ?avoid overfitting?
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) #Uses SGD with the learning rate and the decay to update weights when necessary
model = loadModel("../Data/model/", "model/8")
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
la = model.get_layer(name="convolution3d_3").get_weights()
print(la[0].shape)
print(model.summary())

handler = DataHandler()

#train
for i in range(9):
	print("Training on subset {}".format(i))
	#Loads in the datasets
	xs, ys = handler.load_samples("../Data/sampless/subset{}/".format(i), (1, 10,16,16))
	trainSize =  int(xs.shape[0]*0.2)

	print("training on {} samples".format(trainSize))

	X_train = xs[0:-trainSize]
	y_train = ys[0:-trainSize]
	X_test = xs[-trainSize:]
	y_test = ys[-trainSize:]

	# normalize inputs from 0-255 to 0.0-1.0
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# one hot encode outputs
	#Turns each value into a 1 at the correct position of a vector
	#EX 6 becomes 1 X 10 Vector --> [0,0,0,0,0,1,0,0,0,0]
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)


	#Trains the model on the X and y training data set
	#Uses X_test and y_test to check accuracy while it is training
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
    # Final evaluation of the model
	print(X_test.shape)
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	#saveModel(model, "model/", str(i))
