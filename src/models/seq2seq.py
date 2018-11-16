import numpy as np
import keras
from matplotlib import pyplot as plt
import pandas as pd

#from utils import random_sine, plot_prediction

learning_rate = 0.01
decay = 0 # Learning rate decay

loss = "mse" # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
lambda_regulariser = None #0.000001 # Will not be used if regulariser is None
regulariser = None # Possible regulariser: keras.regularizers.l2(lambda_regulariser)
steps_per_epoch = 200 # batch_size * steps_per_epoch = total number of training examples

num_signals = 2 # The number of random sine waves the compose the signal. The more sine waves, the harder the problem.

def seq2seq_ae(num_input_features, num_output_features, 
               input_sequence_length, target_sequence_length,
              num_steps_to_predict,
              layers=[35, 35]):
    
    keras.backend.clear_session() # clear session/graph    
    optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='encoder_inputs')

    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    # Discard encoder outputs and only keep the states.
    # The outputs are of no interest to us, the encoder's
    # job is to create a state describing the input sequence.
    encoder_states = encoder_outputs_and_states[1:]
    
    # The decoder input will be set to zero (see random_sine function of the utils module).
    # Do not worry about the input size being 1, I will explain that in the next cell.
    decoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='decoder_inputs')

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    # Set the initial state of the decoder to be the ouput state of the encoder.
    # This is the fundamental part of the encoder-decoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    decoder_dense = keras.layers.Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer=regulariser,
                                       bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create a model using the functional API provided by Keras.
    # The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
    # A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)
    
    return model


def seq2seq_pred(num_input_features, num_output_features, num_decoder_features,
               input_sequence_length, target_sequence_length,
              num_steps_to_predict,
              layers=[35, 35]):
    
    keras.backend.clear_session() # clear session/graph    
    optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='encoder_inputs')

    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    # Discard encoder outputs and only keep the states.
    # The outputs are of no interest to us, the encoder's
    # job is to create a state describing the input sequence.
    encoder_states = encoder_outputs_and_states[1:]
    
    # The decoder input will be set to zero (see random_sine function of the utils module).
    # Do not worry about the input size being 1, I will explain that in the next cell.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_features), name='decoder_inputs')

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    # Set the initial state of the decoder to be the ouput state of the encoder.
    # This is the fundamental part of the encoder-decoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    decoder_dense = keras.layers.Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer=regulariser,
                                       bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create a model using the functional API provided by Keras.
    # The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
    # A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)
    
    return model