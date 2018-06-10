import numpy as np
import tensorflow as tf

#TF Imports
from tensorflow.contrib.keras import layers, models, optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import initializers

class Actor:
    
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        l2_regularization_kernel = 1e-5

        # Input Layer
        input = layers.Input(shape=(self.state_size,), name='input_states')

        # Hidden Layers
        model = layers.Dense(units=300, kernel_regularizer=regularizers.l2(l2_regularization_kernel))(input)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(1e-2)(model)

        model = layers.Dense(units=400, kernel_regularizer=regularizers.l2(l2_regularization_kernel))(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(1e-2)(model)

        model = layers.Dense(units=200, kernel_regularizer=regularizers.l2(l2_regularization_kernel))(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(1e-2)(model)

        # Our output layer - a fully connected layer
        output = layers.Dense(units=self.action_size, activation='tanh', kernel_regularizer=regularizers.l2(l2_regularization_kernel),
                               kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3), name='output_actions')(model)

        # Keras model
        self.model = models.Model(inputs=input, outputs=output)

        # Define loss and optimizer
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * output)
        optimizer = optimizers.Adam(lr=1e-4)

        update_operation = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[], updates=update_operation)