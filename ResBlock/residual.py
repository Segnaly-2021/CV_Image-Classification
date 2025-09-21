# Imports
import tensorflow as tf
from tensorflow import keras
from functools import partial

# Create a default Conv2D layer
DefaultConv2D = partial(
    keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False
)

# This decorator tells keras here is a custom object you need to 
# remember when serializing or deserializing

@tf.keras.utils.register_keras_serializable() 
class ResidualBlock(keras.layers.Layer):
    """Creates a custom residual block"""

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        """Builds stacks of layers"""

        # Call the constructors of keras.layers.Layer class
        # **kwargs used for some keyword arguments such as name and dtype
        super().__init__(**kwargs)

        # Set the activation
        self.activation = keras.activations.get(activation)

        # Define a list of layers of a residual architecture
        # block: CL-BN-Relu-CL-BN
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization(),
        ]

        # Define an empty list of layers for inputs to skip the 'normal path'
        self.skip_layers = []

        # Add a stack of CL-BN layers to skip_layers list when two consecutive
        # residual blocks use different strides as it is usually the case
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        """Compute the sum of input and the output of the residual block"""

        # Inputs passing through the "normal path"
        main_output = inputs
        for layer in self.main_layers:
            main_output = layer(main_output)

        # Inputs passing through the "skiping path"
        skip_output = inputs
        for layer in self.skip_layers:
            skip_output = layer(skip_output)

        # summing the main_output and skip_output
        return self.activation(main_output + skip_output)
    
      
    def get_config(self):
        """It allows our residual block to be serializable by keras"""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
            "activation": self.activation_name,
        })
        return config
