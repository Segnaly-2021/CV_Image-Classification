#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ResBlock.residual import ResidualBlock


# In[ ]:


# Load database
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[ ]:


# Create a path for test images
image_test_path = "./images_test"
if not os.path.exists(image_test_path):
    os.mkdir(image_test_path)


def arraytoimage(x, y):
    """Converts numpy array to image"""
    len_x = x.shape[0]
    for i in range(len_x):
        image = Image.fromarray(x[i])
        name = str(y[i])
        image.save(f"{image_test_path }/{name}.png")


# In[ ]:


arraytoimage(x_test, y_test)


# In[ ]:


# Scale test and training set to range from 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[ ]:


# Model definition

model = keras.Sequential()
model.add(keras.layers.Input(shape=(28, 28, 1)))

# Keep the first stack of layers of ResNet with a little change in the
# number of strides, set to 1 instead of 2 (due to the size of images)
model.add(keras.layers.Conv2D(64, kernel_size=7, strides=1, name="label_xf"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same"))

# Number of filters in the first Con2D layer
pre_filter = 64

# Builds the residual blocks using ResidualBlock class
# defined in the ./ResBlock/residual.py file
for filter in [64] * 3 + [128]:
    strides = 1 if filter == pre_filter else 2
    model.add(ResidualBlock(filters=filter, strides=strides))
    pre_filter = filter

# Uses a glabal average layer and then flattens its output
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())

# Add a dense layer and tuner its number of units
model.add(keras.layers.Dense(units=28, activation="relu"))

# Add the output layer
model.add(keras.layers.Dense(10, activation="softmax"))


# In[ ]:


# Adam optimizer and compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.summary()


# In[ ]:


# Train the model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.2,
    steps_per_epoch=1500,
)


# In[ ]:


# Evaluate the model
model.evaluate(x_test, y_test)


# In[ ]:


# Save the model
tf.saved_model.save(model, "./app/models/")


# In[ ]:


# Test
model = tf.saved_model.load("./app/models/")
pred_func = model.signatures["serving_default"]

x_tensor = tf.convert_to_tensor(x_train[0], dtype=tf.float32)

x_tensor = tf.expand_dims(x_tensor, axis=0)
x_tensor = tf.expand_dims(x_tensor, axis=3)

print(np.argmax(pred_func(x_tensor)["output_0"].numpy()))
print(np.max(pred_func(x_tensor)["output_0"].numpy()))

