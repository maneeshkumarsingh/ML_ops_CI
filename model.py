# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt






def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(y_test, y_pred)
  

def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.metrics.mean_squared_error(y_test, y_pred)


# Check Tensorflow version
print(tf.__version__)


# Create features
X = np.arange(-100, 100, 4)

# Create labels
y = np.arange(-90, 110, 4)


# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]


# Take a single example of X
input_shape = X[0].shape 

# Take a single example of y
output_shape = y[0].shape


# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1), 
    tf.keras.layers.Dense(1)
    ])

# Compile the model
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ['mae'])

# Fit the model
model.fit(X_train, y_train, epochs=100)


# Make and plot predictions for model_1
y_preds = model.predict(X_test)
##plot_predictions(train_data=X_train, train_labels=y_train,  test_data=X_test, test_labels=y_test,  predictions=y_preds)


# Calculate model_1 metrics
mae_1 = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
mse_1 = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
