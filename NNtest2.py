import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 hand written images 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer= 'adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


model.save('num_reader.model')

new_model = tf.keras.models.load_model("num_reader.model")

predictions = new_model.predict([x_test])

# print(predictions)

index = 10

import numpy as np
print(np.argmax(predictions[index])) #uses import numpy to take highest probability number in tensor full of predictions to give accurate answer as to which number it is

import matplotlib.pyplot as plt

# plt.imshow(x_train[0], cmap=plt.cm.binary) #shows image 1 at tensor 1 in binary colormap

plt.imshow(x_test[index])
plt.show()
# print(x_train[0])