import tensorflow as tf
print(tf.__version__)

# Check Keras functionality
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(10, input_shape=(10,))])
model.summary()

