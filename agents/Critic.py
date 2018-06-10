import numpy as np
import tensorflow as tf

# Tensorflow layer imports
from tensorflow.contrib.keras import layers, models, optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import initializers

class Critic:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        l2_kernel_regularization = 1e-5

        # Define input layers
        input_states = layers.Input(shape=(self.state_size,), name='input_states')
        input_actions = layers.Input(shape=(self.action_size,), name='input_actions')

        # Hidden layers for states
        model_states = layers.Dense(units=300, kernel_regularizer=regularizers.l2(l2_kernel_regularization))(input_states)
        model_states = layers.BatchNormalization()(model_states)
        model_states = layers.LeakyReLU(1e-2)(model_states)

        model_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(l2_kernel_regularization))(model_states)
        model_states = layers.BatchNormalization()(model_states)
        model_states = layers.LeakyReLU(1e-2)(model_states)

        # Hidden layers for actions
        model_actions = layers.Dense(units=400, kernel_regularizer=regularizers.l2(l2_kernel_regularization))(input_actions)
        model_actions = layers.BatchNormalization()(model_actions)
        model_actions = layers.LeakyReLU(1e-2)(model_actions)

        # Both models merge here
        model = layers.add([model_states, model_actions])

        # Fully connected and batch normalization
        model = layers.Dense(units=200, kernel_regularizer=regularizers.l2(l2_kernel_regularization))(model)
        model = layers.BatchNormalization()(model)
        model = layers.LeakyReLU(1e-2)(model)

        # Q values / output layer
        Q_values = layers.Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(l2_kernel_regularization),
                                kernel_initializer=initializers.RandomUniform(minval=-5e-3, maxval=5e-3),
                                name='output_Q_values')(model)

        # Keras wrap the model
        self.model = models.Model(inputs=[input_states, input_actions], outputs=Q_values)
        optimizer = optimizers.Adam(lr=1e-2)
        self.model.compile(optimizer=optimizer, loss='mse')
        action_gradients = K.gradients(Q_values, input_actions)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

