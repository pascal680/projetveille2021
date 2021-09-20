import tensorflow as tf
from tensorflow import keras

fashiondata = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = fashiondata.load_data()

print(x_test.shape)
print(x_train.shape)

x_train, x_test = x_train/255, x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer = "Adam", 
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy']
                )

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)