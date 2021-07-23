import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2,3)] # petal length, petal width
y = (iris.target == 0).astype(np.int) # Iris setosa

per_clf = Perceptron()
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])

# Implementing MLP with keras

import tensorflow as tf
from tensorflow import keras
# print tensorflow fersion
tf.__version__
### 2.4.1
keras.__version__
### 2.4.0

# classificaiton ANN

# load fashin MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# shape of datasets
X_train_full.shape
X_train_full.dtype

# create validation dataset and scale input approriately for gradient descent
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]

# create feed firward neural network 
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# analog definition

# model = keras.models.Sequential([
#   keras.layers.Flatten(input_shape=[28, 28]),
#   keras.layers.Dense(300, activation="relu"),
#   keras.layers.Dense(100, activation="relu"),
#   keras.layers.Dense(10, activation="softmax")
# ])

# summary of parameters for neural network
model.summary()
model.layers
hidden1 = model.layers[1]
hidden1.name
### dense
model.get_layer("dense") is hidden1

weights, biases = hidden1.get_weights()
weights
weights.shape

biases
biases.shape

# compile the model
model.compile(loss = "sparse_categorical_crossentropy",
optimizer = "sgd",
metrics = ["accuracy"])

# fit = train the compiled model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# visualize results
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set vertical range to [0-1]
plt.show()
plt.clf()


# evaluate model on test dataset
model.evaluate(X_test, y_test)

# predict probability of each class for 3 images
X_new = X_test[:3]
y_probability = model.predict(X_new)
y_probability.round(2)

# directly predict class (having hightest probability)
y_pred = model.predict_classes(X_new)
y_pred
np.array(class_names)[y_pred]

# true labels
y_new = y_test[:3]
y_new



# regression ANN

# import dataset from scikit-learn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
  housing.data, housing.target, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

model = keras.models.Sequential([
  keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
  keras.layers.Dense(1)
])
model.compile(loss = "mean_squared_error", optimizer = "sgd")
history = model.fit(X_train, y_train, epochs = 30,
validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
y_pred

y_new = y_test[:3]
y_new

# building more complex models

input_ = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=3,
validation_data=(X_valid, y_valid))
mset_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
y_pred

# model with different input featues
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape = [6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
validation_data = ((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

# model with two input and two output features
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer="sgd")
history = model.fit(
  [X_train_A, X_train_B], [y_train, y_train], epochs=20,
  validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid])
)

total_loss, main_loss, aux_loss = model.evaluate(
  [X_test_A, X_test_B], [y_test, y_test]
)

y_pred_main, y_pred_aux = model.predict([X_test_A, X_test_B])


# saving and restoring models
# # requires
# model = keras.models.Sequential([...])
# model.compile([...])
# model.fit([...])
model.save("my_keras_model.h5")

# load model
model = keras.models.load_model("my_keras_model.h5")


# using callbacks to save intermediate modeling results
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
save_best_only=True)

input_ = keras.layers.Input(shape = X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss="mse", optimizer="sgd")

history = model.fit(X_train, y_train, epochs=10,
validation_data=(X_valid, y_valid),
callbacks=[checkpoint_cb])

model = keras.models.load_model("my_keras_model.h5") # role back to best model

# early stopping criteria if there is no model improvement
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[checkpoint_cb, early_stopping_cb])

model = keras.models.load_model("my_keras_model.h5")
y_predict = model.predict(X_new)
y_predict

y_new = y_test[:3]
y_new


# TensorBoard for Visualization

import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
  import time
  run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
  return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
validation_data=(X_valid, y_valid),
callbacks=[tensorboard_cb])

# terminal command
# tensorboard --logdir=./my_logs --port=6006



# Fine tuning neural networks hyperparameters

# no support from keras itself --> use sklearn wrapper to do random or grid search
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=input_shape))
  for layer in range(n_hidden):
    model.add(keras.layers.Dense(n_neurons, activation="relu"))
  # add output layer, here for regression
  model.add(keras.layers.Dense(1))
  optimizer = keras.optimizers.SGD(lr=learning_rate)
  model.compile(loss="mse", optimizer=optimizer)
  return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
keras_reg.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)

# random search cross validation
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

params_distribs = {
  "n_hidden": [0, 1, 2, 3],
  "n_neurons": np.arange(1, 100),
  "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, params_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(
  X_train, y_train, epochs=100,
  validation_data=(X_valid, y_valid),
  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
  
rnd_search_cv.best_params_

rnd_search_cv.best_score_

model = rnd_search_cv.best_estimator_.model
model.save("my_keras_model.h5")
# recommended libraries for hyperparameter tuning for ANN with keras and tensorflow:
# Hyperopt, Hyperas, Keras Tuner, Scikit-Optimize, Spearmint (Bayesian Optimization),
# Hyperband, Sklearn-Deap

