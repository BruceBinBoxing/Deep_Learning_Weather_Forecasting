from keras.layers import Dropout, Input, Dense, concatenate, Lambda, Conv1D, BatchNormalization, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model
import tensorflow as tf
from keras import regularizers
import keras
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def mve_loss(y_true, y_pred):
    pred_u = crop(1,0,37)(y_pred)
    pred_sig = crop(1,37,74)(y_pred)
    
    #pred_sig = tf.ones_like(pred_sig)    
    exp_sig = tf.exp(pred_sig) # avoid pred_sig is too small such as zero    
    precision = 1./exp_sig
    log_loss= 0.5*tf.log(exp_sig)+0.5*precision*((pred_u-y_true)**2)
    #log_loss= 0.5*tf.reduce_mean((pred_u-y_true)**2)
    
    log_loss=tf.reduce_mean(log_loss)

    return log_loss

def weather_mve(hidden_nums=100): 
    input_img = Input(shape=(37,))
    hn = Dense(hidden_nums, activation='relu')(input_img)
    hn = Dense(hidden_nums, activation='relu')(hn)
    out_u = Dense(37, activation='sigmoid', name='ae_part')(hn)
    out_sig = Dense(37, activation='linear', name='pred_part')(hn)
    out_both = concatenate([out_u, out_sig], axis=1, name = 'concatenate')

    #weather_model = Model(input_img, outputs=[out_ae, out_pred])
    mve_model = Model(input_img, outputs=[out_both])
    mve_model.compile(optimizer='adam', loss=mve_loss, loss_weights=[1.])
    
    return mve_model

def weather_l2(hidden_nums=100,l2=0.01): 
    input_img = Input(shape=(37,))
    hn = Dense(hidden_nums, activation='relu')(input_img)
    hn = Dense(hidden_nums, activation='relu',
               kernel_regularizer=regularizers.l2(l2))(hn)
    out_u = Dense(37, activation='sigmoid',                 
                  name='ae_part')(hn)
    out_sig = Dense(37, activation='linear', 
                    name='pred_part')(hn)
    out_both = concatenate([out_u, out_sig], axis=1, name = 'concatenate')

    #weather_model = Model(input_img, outputs=[out_ae, out_pred])
    mve_model = Model(input_img, outputs=[out_both])
    mve_model.compile(optimizer='adam', loss=mve_loss, loss_weights=[1.])
    
    return mve_model

def weather_mse():
    input_img = Input(shape=(37,))
    hn = Dense(100, activation='relu')(input_img)
    hn = Dense(100, activation='relu')(hn)
    out_pred = Dense(37, activation='sigmoid', name='pred_part')(hn)
    weather_model = Model(input_img, outputs=[out_pred])
    weather_model.compile(optimizer='adam', loss='mse',loss_weights=[1.])
    
    return weather_model

def RNN_builder(num_output_features, num_decoder_features,
                target_sequence_length,
              num_steps_to_predict, regulariser,
              lr, decay, loss, layers):

    optimiser = keras.optimizers.Adam(lr=lr, decay=decay)
    # Define a decoder sequence.
    decoder_inputs = keras.layers.Input(shape=(37, num_decoder_features), name='decoder_inputs')

    decoder_cells = []

    for hidden_neurons in layers:
        print(hidden_neurons)
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))

    print(decoder_cells)
    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    # Set the initial state of the decoder to be the ouput state of the encoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=None)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    
    #decoder_dense1 = keras.layers.Dense(units=64,
    #                                   activation='tanh',
    #                                   kernel_regularizer = regulariser,
    #                                   bias_regularizer = regulariser, name='dense_tanh')

    output_dense = keras.layers.Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_sig')

    #densen1=decoder_dense1(decoder_outputs)
    decoder_outputs = output_dense(decoder_outputs)
    # Create a model using the functional API provided by Keras.
    rnn_model = keras.models.Model(inputs=[decoder_inputs], outputs=decoder_outputs)
    return rnn_model

def Seq2Seq(id_embd, time_embd,
            lr, decay,
            num_input_features, num_output_features,
            num_decoder_features, layers,
            loss, regulariser):
    optimiser = keras.optimizers.Adam(lr=lr, decay=decay)
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='encoder_inputs')
    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))
        
    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    # Discard encoder outputs and only keep the states.
    encoder_states = encoder_outputs_and_states[1:]
    # Define a decoder sequence.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_features), name='decoder_inputs')
    
    decoder_inputs_id = keras.layers.Input(shape=(None,), name='id_inputs')
    decoder_inputs_id_embd = Embedding(input_dim=10, output_dim=2, name='id_embedding')(decoder_inputs_id)

    decoder_inputs_time = keras.layers.Input(shape=(None,), name='time_inputs')
    decoder_inputs_time_embd = Embedding(input_dim=37, output_dim=2, name='time_embedding')(decoder_inputs_time)

    if id_embd and (not time_embd):
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd], axis=-1, name='concat_inputs_id')
    elif id_embd and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd, decoder_inputs_time_embd], axis=-1, name='concat_inputs_id_time')
    elif (not id_embd) and (not time_embd):
        decoder_concat = decoder_inputs
    elif (not id_embd) and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_time_embd], axis=-1, name='concat_inputs_time')

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder(decoder_concat, initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]
    output_dense = keras.layers.Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_sig')

    decoder_outputs = output_dense(decoder_outputs)

    if id_embd and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id], outputs=decoder_outputs)
    elif id_embd and time_embd:
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id, decoder_inputs_time], outputs=decoder_outputs)
    elif (not id_embd) and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    return model

def Seq2Seq_MVE(id_embd, time_embd,
            lr, decay,
            num_input_features, num_output_features,
            num_decoder_features, layers,
            loss, regulariser, dropout_rate):
    optimiser = keras.optimizers.Adam(lr=lr, decay=decay)
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='encoder_inputs')
    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))
        
    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    # Discard encoder outputs and only keep the states.
    encoder_states = encoder_outputs_and_states[1:]
    # Define a decoder sequence.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_features), name='decoder_inputs')
    
    decoder_inputs_id = keras.layers.Input(shape=(None,), name='id_inputs')
    decoder_inputs_id_embd = Embedding(input_dim=10, output_dim=2, name='id_embedding')(decoder_inputs_id)

    decoder_inputs_time = keras.layers.Input(shape=(None,), name='time_inputs')
    decoder_inputs_time_embd = Embedding(input_dim=37, output_dim=2, name='time_embedding')(decoder_inputs_time)

    if id_embd and (not time_embd):
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd], axis=-1, name='concat_inputs_id')
    elif id_embd and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd, decoder_inputs_time_embd], axis=-1, name='concat_inputs_id_time')
    elif (not id_embd) and (not time_embd):
        decoder_concat = decoder_inputs
    elif (not id_embd) and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_time_embd], axis=-1, name='concat_inputs_time')

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder(decoder_concat, initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]

    output_dense = keras.layers.Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_mean')

    variance_dense = keras.layers.Dense(num_output_features,
                                       activation='softplus',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_variance')

    mean_outputs = output_dense(decoder_outputs)
    variance_outputs = variance_dense(decoder_outputs)

    mve_outputs = concatenate([mean_outputs, variance_outputs], axis=-1, name = 'output_mve')

    if id_embd and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id], outputs=[mean_outputs])
    elif id_embd and time_embd:
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id, decoder_inputs_time], outputs=[mve_outputs])
    elif (not id_embd) and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[mean_outputs])

    return model

def Seq2Seq_MVE_subnets(id_embd, time_embd,
            lr, decay,
            num_input_features, num_output_features,
            num_decoder_features, layers,
            loss, regulariser):
    optimiser = keras.optimizers.Adam(lr=lr, decay=decay)
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='encoder_inputs')
    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))
        
    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    # Discard encoder outputs and only keep the states.
    encoder_states = encoder_outputs_and_states[1:]
    # Define a decoder sequence.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_features), name='decoder_inputs')
    
    decoder_inputs_id = keras.layers.Input(shape=(None,), name='id_inputs')
    decoder_inputs_id_embd = Embedding(input_dim=10, output_dim=2, name='id_embedding')(decoder_inputs_id)

    decoder_inputs_time = keras.layers.Input(shape=(None,), name='time_inputs')
    decoder_inputs_time_embd = Embedding(input_dim=37, output_dim=2, name='time_embedding')(decoder_inputs_time)

    if id_embd and (not time_embd):
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd], axis=-1, name='concat_inputs_id')
    elif id_embd and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd, decoder_inputs_time_embd], axis=-1, name='concat_inputs_id_time')
    elif (not id_embd) and (not time_embd):
        decoder_concat = decoder_inputs
    elif (not id_embd) and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_time_embd], axis=-1, name='concat_inputs_time')

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder(decoder_concat, initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]

    mean_outputs = Dense(30, activation='tanh', name='subnet_mean')(decoder_outputs)
    mean_outputs = Dropout(rate=0.3, name='subnet_mean_drop_layer')(mean_outputs)

    variance_outputs = Dense(30, activation='tanh', name='subnet_var')(decoder_outputs)
    variance_outputs = Dropout(rate=0.3, name='subnet_var_drop_layer')(variance_outputs)

    mean_outputs = Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_mean')(mean_outputs)
    variance_outputs = Dense(num_output_features,
                                       activation='softplus',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_variance')(variance_outputs)

    mve_outputs = concatenate([mean_outputs, variance_outputs], axis=-1, name = 'output_mve')

    if id_embd and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id], outputs=[mean_outputs])
    elif id_embd and time_embd:
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id, decoder_inputs_time], outputs=[mve_outputs])
    elif (not id_embd) and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[mean_outputs])

    return model

def Seq2Seq_MVE_subnets_swish(id_embd, time_embd,
            lr, decay,
            num_input_features, num_output_features,
            num_decoder_features, layers,
            loss, regulariser, dropout_rate):
    
    def swish(x):
        return (K.sigmoid(x) * x)
    
    get_custom_objects().update({'swish':swish})

    optimiser = keras.optimizers.Adam(lr=lr, decay=decay)
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features), name='encoder_inputs')
    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))
        
    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    # Discard encoder outputs and only keep the states.
    encoder_states = encoder_outputs_and_states[1:]
    # Define a decoder sequence.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_features), name='decoder_inputs')
    
    decoder_inputs_id = keras.layers.Input(shape=(None,), name='id_inputs')
    decoder_inputs_id_embd = Embedding(input_dim=10, output_dim=2, name='id_embedding')(decoder_inputs_id)

    decoder_inputs_time = keras.layers.Input(shape=(None,), name='time_inputs')
    decoder_inputs_time_embd = Embedding(input_dim=37, output_dim=2, name='time_embedding')(decoder_inputs_time)

    if id_embd and (not time_embd):
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd], axis=-1, name='concat_inputs_id')
    elif id_embd and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd, decoder_inputs_time_embd], axis=-1, name='concat_inputs_id_time')
    elif (not id_embd) and (not time_embd):
        decoder_concat = decoder_inputs
    elif (not id_embd) and time_embd:
        decoder_concat = concatenate([decoder_inputs, decoder_inputs_time_embd], axis=-1, name='concat_inputs_time')

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer = regulariser,
                                                  recurrent_regularizer = regulariser,
                                                  bias_regularizer = regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder(decoder_concat, initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]

    mean_outputs = Dense(10, activation='swish')(decoder_outputs)
    mean_outputs = Dropout(rate=dropout_rate, name='subnet_mean_drop_layer')(mean_outputs)

    variance_outputs = Dense(10, activation='swish')(decoder_outputs)
    variance_outputs = Dropout(rate=dropout_rate, name='subnet_var_drop_layer')(variance_outputs)

    mean_outputs = Dense(num_output_features,
                                       activation='sigmoid',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_mean')(mean_outputs)
    variance_outputs = Dense(num_output_features,
                                       activation='softplus',
                                       kernel_regularizer = regulariser,
                                       bias_regularizer = regulariser, name='output_variance')(variance_outputs)

    mve_outputs = concatenate([mean_outputs, variance_outputs], axis=-1, name = 'output_mve')

    if id_embd and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id], outputs=[mean_outputs])
    elif id_embd and time_embd:
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id, decoder_inputs_time], outputs=[mve_outputs])
    elif (not id_embd) and (not time_embd):
        model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[mean_outputs])

    return model

def CausalCNN(n_filters, lr, decay, loss, 
               seq_len, input_features, 
               strides_len, kernel_size,
               dilation_rates):

    inputs = Input(shape=(seq_len, input_features), name='input_layer')   
    x=inputs
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
               kernel_size=kernel_size, 
               padding='causal',
               dilation_rate=dilation_rate,
               activation='linear')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    #x = Dense(7, activation='relu', name='dense_layer')(x)
    outputs = Dense(3, activation='sigmoid', name='output_layer')(x)
    causalcnn = Model(inputs, outputs=[outputs])

    return causalcnn

def weather_conv1D(layers, lr, decay, loss, 
               input_len, input_features, 
               strides_len, kernel_size):
    
    inputs = Input(shape=(input_len, input_features), name='input_layer')
    for i, hidden_nums in enumerate(layers):
        if i==0:
            #inputs = BatchNormalization(name='BN_input')(inputs)
            hn = Conv1D(hidden_nums, kernel_size=kernel_size, strides=strides_len, 
                        data_format='channels_last', 
                        padding='same', activation='linear')(inputs)
            hn = BatchNormalization(name='BN_{}'.format(i))(hn)
            hn = Activation('relu')(hn)
        elif i<len(layers)-1:
            hn = Conv1D(hidden_nums, kernel_size=kernel_size, strides=strides_len,
                        data_format='channels_last', 
                        padding='same',activation='linear')(hn)
            hn = BatchNormalization(name='BN_{}'.format(i))(hn) 
            hn = Activation('relu')(hn)
        else:
            hn = Conv1D(hidden_nums, kernel_size=kernel_size, strides=strides_len,
                        data_format='channels_last', 
                        padding='same',activation='linear')(hn)
            hn = BatchNormalization(name='BN_{}'.format(i))(hn) 

    outputs = Dense(80, activation='relu', name='dense_layer')(hn)
    outputs = Dense(3, activation='tanh', name='output_layer')(outputs)

    weather_model = Model(inputs, outputs=[outputs])

    return weather_model

def weather_fnn(layers, lr,
            decay, loss, seq_len, 
            input_features, output_features):
    
    ori_inputs = Input(shape=(seq_len, input_features), name='input_layer')
    #print(seq_len*input_features)
    conv_ = Conv1D(11, kernel_size=13, strides=1, 
                        data_format='channels_last', 
                        padding='valid', activation='linear')(ori_inputs)
    conv_ = BatchNormalization(name='BN_conv')(conv_)
    conv_ = Activation('relu')(conv_)
    conv_ = Conv1D(5, kernel_size=7, strides=1, 
                        data_format='channels_last', 
                        padding='valid', activation='linear')(conv_)
    conv_ = BatchNormalization(name='BN_conv2')(conv_)
    conv_ = Activation('relu')(conv_)

    inputs = Reshape((-1,))(conv_)

    for i, hidden_nums in enumerate(layers):
        if i==0:
            hn = Dense(hidden_nums, activation='linear')(inputs)
            hn = BatchNormalization(name='BN_{}'.format(i))(hn)
            hn = Activation('relu')(hn)
        else:
            hn = Dense(hidden_nums, activation='linear')(hn)
            hn = BatchNormalization(name='BN_{}'.format(i))(hn)
            hn = Activation('relu')(hn)
            #hn = Dropout(0.1)(hn)
    #print(seq_len, output_features)
    #print(hn)
    outputs = Dense(seq_len*output_features, activation='sigmoid', name='output_layer')(hn) # 37*3
    outputs = Reshape((seq_len, output_features))(outputs)

    weather_fnn = Model(ori_inputs, outputs=[outputs])

    return weather_fnn

def weather_ae(layers, lr, decay, loss, 
               input_len, input_features):
    
    inputs = Input(shape=(input_len, input_features), name='input_layer')
    
    for i, hidden_nums in enumerate(layers):
        if i==0:
            hn = Dense(hidden_nums, activation='relu')(inputs)
        else:
            hn = Dense(hidden_nums, activation='relu')(hn)

    outputs = Dense(3, activation='sigmoid', name='output_layer')(hn)

    weather_model = Model(inputs, outputs=[outputs])

    return weather_model

def weather_fusion():
    input_img = Input(shape=(37,))
    hn = Dense(100, activation='relu')(input_img)
    hn = Dense(100, activation='relu')(hn)
    #out_ae = Dense(37, activation='sigmoid', name='ae_part')(hn)
    out_pred = Dense(37, activation='sigmoid', name='pred_part')(hn)

    weather_model = Model(input_img, outputs=[out_ae, out_pred])
    weather_model.compile(optimizer='adam', loss='mse',loss_weights=[1.5, 1.])

    return weather_model
    