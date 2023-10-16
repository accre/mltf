#!/usr/bin/env python3

import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt

mlflow.autolog()

def plot_save_dig(image_data, fig_name):
   plt.imshow(image_data, cmap="binary")
   plt.axis("off")
   plt.savefig(fig_name)

mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

X_train, X_test, X_valid = X_train/255., X_test/255., X_valid/255.

model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=[28, 28]),
	tf.keras.layers.Dense(200, activation="sigmoid"),
	tf.keras.layers.Dense(200, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="softmax")
	])

print(model.summary())
	
model.compile(loss="sparse_categorical_crossentropy",
	optimizer="sgd",
	metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs = 20,
		validation_data=(X_valid, y_valid))

model.evaluate(X_test, y_test)

#plot_save_dig(X_train_full[1], "first_digit")



