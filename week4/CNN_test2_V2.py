# keras imports for the dataset and building our neural network
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import tensorflow as tf



# loading the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(tf.__version__)
# # building the input vector from the 32x32 pixels
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalizing the data to help with the training
x_train /= 255
x_test /= 255

# one-hot encoding using keras' numpy-related utilities
nb_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print("Shape after one-hot encoding: ", y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()


# convolutional layer
model.add(Conv2D(70, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns
model.add(MaxPool2D(pool_size=(2,2))) #Compress filtered data for image and reinforce patterns
model.add(Dropout(0.25)) #Remove useless filtered images and reduce training time

model.add(Conv2D(70, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns
model.add(MaxPool2D(pool_size=(2,2))) #Compress filtered data for image and reinforce patterns
model.add(Dropout(0.25)) #Remove useless filtered images and reduce training time

# flatten output of conv
model.add(Flatten()) #Flattens all tensor with all the 

# hidden layer
model.add(Dense(400, activation='relu')) #Learning from the reinforced patterns in the convolutional layers
model.add(Dropout(0.4)) #Heavy dropout rate to increase training speed and drop the useless neurons 
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(nb_classes, activation='softmax')) #Outputs the category in which the image belongs too according to the current state of the CNN

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_test, y_test))

model.save('image_categorizing_cnn.model')

