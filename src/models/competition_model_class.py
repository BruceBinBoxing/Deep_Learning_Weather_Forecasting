import numpy as np
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Lambda
import os, sys

from weather_model import Seq2Seq_MVE_subnets_swish, weather_conv1D, CausalCNN, RNN_builder, Seq2Seq, Seq2Seq_MVE, Seq2Seq_MVE_subnets
from keras.models import load_model, model_from_json

#from utils import random_sine, plot_prediction

#learning_rate = 0.01
#decay = 0 # Learning rate decay
model_save_path = '../models/'
#loss = "mse" # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
#lambda_regulariser = None #0.000001 # Will not be used if regulariser is None
#regulariser = None # Possible regulariser: keras.regularizers.l2(lambda_regulariser)
#steps_per_epoch = 200 # batch_size * steps_per_epoch = total number of training examples
#num_signals = 2 # The number of random sine waves the compose the signal. The more sine waves, the harder the problem.


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

class WeatherConv1D:
    def __init__(self, regulariser=None,
              lr=0.001, decay=0, loss="mse",
              layers=[35, 35], batch_size=256, 
              input_len=37, input_features=29, 
              strides_len=3, kernel_size=5):      
        self.regulariser = regulariser
        self.layers = layers
        self.lr = lr
        self.decay = decay
        self.loss = loss
        self.pred_result = None
        self.batch_size = batch_size
        self.input_len = input_len
        self.input_features = input_features
        self.kernel_strides = strides_len
        self.kernel_size = kernel_size

        print('Initialized!')

    def build_graph(self):
        keras.backend.clear_session() # clear session/graph

        self.model = weather_conv1D(self.layers, self.lr,
            self.decay, self.loss, self.input_len, 
            self.input_features, self.kernel_strides, self.kernel_size)

        print(self.model.summary())
        
    def sample_batch(self, data_inputs, ground_truth, ruitu_inputs, batch_size, certain_id=None, certain_feature=None):
            
            max_i, _, max_j, _ = data_inputs.shape # Example: (1148, 37, 10, 9)-(sample_ind, timestep, sta_id, features)
            
            if certain_id == None and certain_feature == None:
                id_ = np.random.randint(max_j, size=batch_size)
                i = np.random.randint(max_i, size=batch_size)

                batch_ouputs = ground_truth[i,:,id_,:]
                batch_ruitu = ruitu_inputs[i,:,id_,:]

            elif certain_id != None:
                pass

            return batch_ruitu, batch_ouputs

    def order_batch(self, data_inputs, ground_truth, ruitu_inputs, batch_size, certain_id=None, certain_feature=None):
            pass #TODO:

    def fit(self, train_input_ruitu, train_labels,
            val_input_ruitu, val_labels, batch_size,
            iterations=300, validation=True):
                
        self.optimizer = keras.optimizers.Adam(lr=self.lr, decay=self.decay)
        self.model.compile(optimizer = self.optimizer, loss=self.loss)
        
        print('Train batch size: {}'.format(batch_size))
        print('Validation on data size of {};'.format(val_input_ruitu.shape[0]))
        
        for i in range(iterations):
            batch_ruitu, batch_labels = self.sample_batch(train_input_ruitu, train_labels, 
                                                                         train_input_ruitu, batch_size=batch_size)
            loss_ = self.model.train_on_batch(x=[batch_ruitu], 
                  y=[batch_labels])

            if (i+1)%50 == 0:
                print('Iteration:{}/{}. Training batch loss:{}'.
                      format(i+1, iterations, loss_))
                if validation :
                    self.evaluate(val_input_ruitu, val_labels, each_station_display=False)                
        


        print('###'*10)
        print('Train finish! Total validation loss:')
        self.evaluate(val_input_ruitu, val_labels, each_station_display=True)

    def evaluate(self, data_input_ruitu, data_labels, each_station_display=False):       
        all_loss=[]
        for i in range(10): # iterate for each station. (sample_ind, timestep, staionID, features) 
            val_loss= self.model.evaluate(x=[data_input_ruitu[:,:,i,:]],
                                y=[data_labels[:,:,i,:]], verbose=False)

            all_loss.append(val_loss)

            if each_station_display:
                print('\tFor station 9000{}, evaluated loss: {}'.format(i+1, val_loss))
        
        print('Mean evaluated loss on all stations:', np.mean(all_loss))

        #return np.mean(all_loss)

    def predict(self, batch_ruitu):
        pred_result_list = []
        for i in range(10):
            #print('Predict for station: 9000{}'.format(i+1))
            result = self.model.predict(x=[batch_ruitu[:,:,i,:]])
            result = np.squeeze(result, axis=0)
            #all_pred[i] = result
            pred_result_list.append(result)
            #pass

        pred_result = np.stack(pred_result_list, axis=0)
        #return all_pred, pred_result
        print('Predict shape (10,37,3) means (stationID, timestep, features). Features include: t2m, rh2m and w10m')
        self.pred_result = pred_result
        return pred_result

    def renorm_for_submit(self, pred_mean, pred_var):
        if self.pred_result is None:
            print('You must run self.predict(batch_inputs, batch_ruitu) firstly!!')
        else:
            df_empty = pd.DataFrame(columns=['FORE_data', 't2m', 'rh2m', 'w10m'])

            target_list=['t2m','rh2m','w10m']

            self.obs_range_dic={'t2m':[-30,42], # Official value: [-20,42]
                             'rh2m':[0.0,100.0],
                             'w10m':[0.0, 30.0]}

            for j, target_v in enumerate(self.target_list):
                
                series_ids = pd.Series()
                series_targets = pd.Series()
                
                renorm_value = renorm(self.pred_result[:,:,j], self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
                    
                for i in range(10):
                    if i != 9:
                        id_num = '0'+str(i+1)
                    else:
                        id_num = str(10)
                    sta_name_time = '900'+id_num+'_'
                    
                    time_str_list=[]

                    for t in range(37):
                        if t < 10:
                            time_str= sta_name_time + '0'+ str(t)
                        else:
                            time_str = sta_name_time + str(t)
                        
                        time_str_list.append(time_str)
                    
                    series_id = pd.Series(time_str_list)
                    series_target = pd.Series(renorm_value[i])
                    
                    series_ids = pd.concat([series_ids, series_id])
                    series_targets = pd.concat([series_targets, series_target])
                    
                df_empty['FORE_data'] = series_ids
                df_empty[target_v] = series_targets

            return df_empty

class CausalCNN_Class(WeatherConv1D):
    def __init__(self, regulariser,lr, decay, loss, 
        n_filters, strides_len, kernel_size, seq_len,
        input_features, output_features, dilation_rates):

        self.regulariser=regulariser
        self.n_filters=n_filters
        self.lr=lr
        self.decay=decay
        self.loss=loss
        self.seq_len=seq_len
        self.input_features=input_features
        self.output_features = output_features
        self.strides_len=strides_len
        self.kernel_size=kernel_size
        self.dilation_rates=dilation_rates


    def build_graph(self):
        keras.backend.clear_session() # clear session/graph

        self.model = CausalCNN(self.n_filters, self.lr, 
            self.decay, self.loss, 
               self.seq_len, self.input_features, 
               self.strides_len, self.kernel_size,
               self.dilation_rates)

        print(self.model.summary())
         
class FNN(WeatherConv1D):

    def __init__(self, regulariser,lr, decay, loss, 
        layers, batch_size, seq_len, input_features, output_features):

        self.regulariser=regulariser
        self.layers=layers
        self.lr=lr
        self.decay=decay
        self.loss=loss
        self.seq_len=seq_len
        self.input_features=input_features
        self.output_features = output_features

    def build_graph(self):
        keras.backend.clear_session() # clear session/graph

        self.model = weather_fnn(self.layers, self.lr,
            self.decay, self.loss, self.seq_len, 
            self.input_features, self.output_features)

        print(self.model.summary())



class Enc_Dec:
    def __init__(self, num_input_features, num_output_features, num_decoder_features,
               input_sequence_length, target_sequence_length,
              num_steps_to_predict, regulariser = None,
              lr=0.001, decay=0, loss = "mse",
              layers=[35, 35]):
        
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.num_decoder_features = num_decoder_features
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.num_steps_to_predict = num_steps_to_predict
        self.regulariser = regulariser
        self.layers = layers
        self.lr = lr
        self.decay = decay
        self.loss = loss
        self.pred_result = None
        self.train_loss=[]

        self.target_list=['t2m','rh2m','w10m']

        self.obs_range_dic={'t2m':[-30,42], # Official value: [-20,42]
                         'rh2m':[0.0,100.0],
                         'w10m':[0.0, 30.0]}

        print('Initialized!')

    def build_graph(self):
        keras.backend.clear_session() # clear session/graph    
        self.optimiser = keras.optimizers.Adam(lr=self.lr, decay=self.decay)
        # Define an input sequence.
        encoder_inputs = keras.layers.Input(shape=(None, self.num_input_features), name='encoder_inputs')
        # Create a list of RNN Cells, these are then concatenated into a single layer
        # with the RNN layer.
        encoder_cells = []
        for hidden_neurons in self.layers:
            encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                      kernel_regularizer = self.regulariser,
                                                      recurrent_regularizer = self.regulariser,
                                                      bias_regularizer = self.regulariser))
            
        encoder = keras.layers.RNN(encoder_cells, return_state=True)
        encoder_outputs_and_states = encoder(encoder_inputs)
        # Discard encoder outputs and only keep the states.
        encoder_states = encoder_outputs_and_states[1:]
        # Define a decoder sequence.
        decoder_inputs = keras.layers.Input(shape=(None, self.num_decoder_features), name='decoder_inputs')

        decoder_cells = []
        for hidden_neurons in self.layers:
            decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                      kernel_regularizer = self.regulariser,
                                                      recurrent_regularizer = self.regulariser,
                                                      bias_regularizer = self.regulariser))

        decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
        # Set the initial state of the decoder to be the ouput state of the encoder.
        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

        # Only select the output of the decoder (not the states)
        decoder_outputs = decoder_outputs_and_states[0]

        # Apply a dense layer with linear activation to set output to correct dimension
        # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
        
        decoder_dense1 = keras.layers.Dense(units=64,
                                           activation='tanh',
                                           kernel_regularizer = self.regulariser,
                                           bias_regularizer = self.regulariser, name='dense_tanh')

        output_dense = keras.layers.Dense(self.num_output_features,
                                           activation='sigmoid',
                                           kernel_regularizer = self.regulariser,
                                           bias_regularizer = self.regulariser, name='output_sig')

        #densen1=decoder_dense1(decoder_outputs)
        decoder_outputs = output_dense(decoder_outputs)
        # Create a model using the functional API provided by Keras.
        self.model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        print(self.model.summary())
        
    def sample_batch(self, data_inputs, ground_truth, ruitu_inputs, batch_size, certain_id=None, certain_feature=None):
            
            max_i, _, max_j, _ = data_inputs.shape # Example: (1148, 37, 10, 9)-(sample_ind, timestep, sta_id, features)
            
            if certain_id == None and certain_feature == None:
                id_ = np.random.randint(max_j, size=batch_size)
                i = np.random.randint(max_i, size=batch_size)

                batch_inputs = data_inputs[i,:,id_,:]
                batch_ouputs = ground_truth[i,:,id_,:]
                batch_ruitu = ruitu_inputs[i,:,id_,:]

            elif certain_id != None:
                pass

            return batch_inputs, batch_ruitu, batch_ouputs

    def fit(self, train_input_obs, train_input_ruitu, train_labels,
            val_input_obs, val_input_ruitu, val_labels, batch_size,
            iterations=300, validation=True):
        self.model.compile(optimizer = self.optimiser, loss=self.loss)
        
        print('Train batch size: {}'.format(batch_size))
        print('Validation on data size of {};'.format(val_input_obs.shape[0]))
        
        for i in range(iterations):
            batch_inputs, batch_ruitu, batch_labels = self.sample_batch(train_input_obs, train_labels, 
                                                                         train_input_ruitu, batch_size=batch_size)
            loss_ = self.model.train_on_batch(x=[batch_inputs, batch_ruitu], 
                  y=[batch_labels])

            if (i+1)%50 == 0:
                print('Iteration:{}/{}. Training batch loss:{}'.
                      format(i+1, iterations, loss_))
                if validation :
                    self.evaluate(val_input_obs, val_input_ruitu, val_labels, each_station_display=False)                

        print('###'*10)
        print('Train finish! Total validation loss:')
        self.evaluate(val_input_obs, val_input_ruitu, val_labels, each_station_display=True)

    def evaluate(self, data_input_obs, data_input_ruitu, data_labels, each_station_display=False):
        assert data_input_ruitu.shape[0] == data_input_obs.shape[0] == data_labels.shape[0], 'Shape Error'
        #assert data_input_obs.shape[1] == 28 and data_input_obs.shape[2] == 10 and data_input_obs.shape[3] == 9, 'Error! Obs input shape must be (None, 28,10,9)'
        assert data_input_ruitu.shape[1] == 37 and data_input_ruitu.shape[2] == 10 and data_input_ruitu.shape[3] == 29, 'Error! Ruitu input shape must be (None, 37,10,29)'
        assert data_labels.shape[1] == 37 and data_labels.shape[2] == 10 and data_labels.shape[3] == 3, 'Error! Ruitu input shape must be (None, 37,10,3)' 
        
        all_loss=[]

        for i in range(10): # iterate for each station. (sample_ind, timestep, staionID, features) 
            val_loss= self.model.evaluate(x=[data_input_obs[:,:,i,:], data_input_ruitu[:,:,i,:]],
                                y=[data_labels[:,:,i,:]], verbose=False)

            all_loss.append(val_loss)

            if each_station_display:
                print('\tFor station 9000{}, evaluated loss: {}'.format(i+1, val_loss))
        
        print('Mean evaluated loss on all stations:', np.mean(all_loss))

    def predict(self, batch_inputs, batch_ruitu):
        assert batch_ruitu.shape[0] == batch_inputs.shape[0], 'Shape Error'
        assert batch_inputs.shape[1] == 28 and batch_inputs.shape[2] == 10 and batch_inputs.shape[3] == 9, 'Error! Obs input shape must be (None, 28,10,9)'
        assert batch_ruitu.shape[1] == 37 and batch_ruitu.shape[2] == 10 and batch_ruitu.shape[3] == 29, 'Error! Ruitu input shape must be (None, 37,10, 29)'
        #all_pred={}
        pred_result_list = []
        for i in range(10):
            #print('Predict for station: 9000{}'.format(i+1))
            result = self.model.predict(x=[batch_inputs[:,:,i,:], batch_ruitu[:,:,i,:]])
            result = np.squeeze(result, axis=0)
            #all_pred[i] = result
            pred_result_list.append(result)
            #pass

        pred_result = np.stack(pred_result_list, axis=0)
        #return all_pred, pred_result
        print('Predict shape (10,37,3) means (stationID, timestep, features). Features include: t2m, rh2m and w10m')
        self.pred_result = pred_result
        return pred_result

    def renorm_for_submit(self, pred_mean, pred_var=None):
        '''
        # TODO: Add three strategies for output

        '''
        assert self.pred_result is not None, 'You must run self.predict(batch_inputs, batch_ruitu) firstly!!'
        assert pred_mean.shape == (10, 37, 3), 'Error! This funtion ONLY works for one data sample with shape (10, 37, 3). Any data shape (None, 10, 37, 3) will leads this error!'

        df_empty = pd.DataFrame(columns=['FORE_data', 't2m', 'rh2m', 'w10m'])

        for j, target_v in enumerate(self.target_list):
            
            series_ids = pd.Series()
            series_targets = pd.Series()
            
            renorm_value = renorm(pred_mean[:,:,j], self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
                
            for i in range(10):
                if i != 9:
                    id_num = '0'+str(i+1)
                else:
                    id_num = str(10)
                sta_name_time = '900'+id_num+'_'
                
                time_str_list=[]

                for t in range(37):
                    if t < 10:
                        time_str= sta_name_time + '0'+ str(t)
                    else:
                        time_str = sta_name_time + str(t)
                    
                    time_str_list.append(time_str)
                
                series_id = pd.Series(time_str_list)
                series_target = pd.Series(renorm_value[i])
                
                series_ids = pd.concat([series_ids, series_id])
                series_targets = pd.concat([series_targets, series_target])
                
            df_empty['FORE_data'] = series_ids
            df_empty[target_v] = series_targets

        return df_empty
        #pass
    
    def plot_prediction(self, x, y_true, y_pred, input_ruitu=None):
        """Plots the predictions.

        Arguments
        ---------
        x: Input sequence of shape (input_sequence_length,
            dimension_of_signal)
        y_true: True output sequence of shape (input_sequence_length,
            dimension_of_signal)
        y_pred: Predicted output sequence (input_sequence_length,
            dimension_of_signal)
        input_ruitu: Ruitu output sequence 
        """

        plt.figure(figsize=(12, 3))

        output_dim = x.shape[-1]# feature dimension
        for j in range(output_dim):
            past = x[:, j] 
            true = y_true[:, j]
            pred = y_pred[:, j]
            if input_ruitu is not None:
                ruitu = input_ruitu[:, j]

            label1 = "Seen (past) values" if j==0 else "_nolegend_"
            label2 = "True future values" if j==0 else "_nolegend_"
            label3 = "Predictions" if j==0 else "_nolegend_"
            label4 = "Ruitu values" if j==0 else "_nolegend_"

            plt.plot(range(len(past)), past, "o-g",
                     label=label1)
            plt.plot(range(len(past),
                     len(true)+len(past)), true, "x--g", label=label2)
            plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                     label=label3)
            if input_ruitu is not None:
                plt.plot(range(len(past), len(ruitu)+len(past)), ruitu, "o--r",
                     label=label4)
        plt.legend(loc='best')
        plt.title("Predictions v.s. true values v.s. Ruitu")
        plt.show()

class RNN_Class(WeatherConv1D):
    def __init__(self, num_output_features, num_decoder_features,
                target_sequence_length,
              num_steps_to_predict, regulariser = None,
              lr=0.001, decay=0, loss = "mse",
              layers=[35, 35]):

        self.num_output_features = num_output_features
        self.num_decoder_features = num_decoder_features
        self.target_sequence_length = target_sequence_length
        self.num_steps_to_predict = num_steps_to_predict
        self.regulariser = regulariser
        self.layers = layers
        self.lr = lr
        self.decay = decay
        self.loss = loss
        self.pred_result = None
        #self.batch_size = batch_size
        print('Initialized!')

    def build_graph(self):
        keras.backend.clear_session() # clear session/graph
        self.model = RNN_builder(self.num_output_features, self.num_decoder_features,
                self.target_sequence_length,
              self.num_steps_to_predict, self.regulariser,
              self.lr, self.decay, self.loss, self.layers)

        print(self.model.summary())

class Seq2Seq_Class(Enc_Dec):
    def __init__(self, id_embd, time_embd, 
        num_input_features, num_output_features, num_decoder_features,
               input_sequence_length, target_sequence_length,
              num_steps_to_predict, regulariser = None,
              lr=0.001, decay=0, loss = "mse",
              layers=[35, 35], model_save_path='../models', 
              model_structure_name='seq2seq_model.json', model_weights_name='seq2seq_model_weights.h5'):
        
        super().__init__(num_input_features, num_output_features, num_decoder_features,
               input_sequence_length, target_sequence_length,
              num_steps_to_predict, regulariser = None,
              lr=lr, decay=decay, loss = loss,
              layers=layers)

        self.id_embd = id_embd
        self.time_embd = time_embd
        self.val_loss_list=[]
        self.train_loss_list=[]
        self.current_mean_val_loss = None
        self.early_stop_limit = 10 # with the unit of Iteration Display
        self.EARLY_STOP=False
        self.pred_var_result = []

        self.pi_dic={0.95:1.96, 0.9:1.645, 0.8:1.28, 0.68:1.}
        self.target_list=['t2m','rh2m','w10m']
        self.obs_range_dic={'t2m':[-30,42], # Official value: [-20,42]
                         'rh2m':[0.0,100.0],
                         'w10m':[0.0, 30.0]}
        self.obs_and_output_feature_index_map = {'t2m':0,'rh2m':1,'w10m':2}
        self.ruitu_feature_index_map = {'t2m':1,'rh2m':3,'w10m':4}
        self.model_save_path = model_save_path
        self.model_structure_name=model_structure_name
        self.model_weights_name=model_weights_name

    def build_graph(self):
        #keras.backend.clear_session() # clear session/graph    
        self.optimizer = keras.optimizers.Adam(lr=self.lr, decay=self.decay)

        self.model = Seq2Seq_MVE_subnets_swish(id_embd=True, time_embd=True,
            lr=self.lr, decay=self.decay,
            num_input_features=self.num_input_features, num_output_features=self.num_output_features,
            num_decoder_features=self.num_decoder_features, layers=self.layers,
            loss=self.loss, regulariser=self.regulariser)

        def _mve_loss(y_true, y_pred):
            pred_u = crop(2,0,3)(y_pred)
            pred_sig = crop(2,3,6)(y_pred)
            print(pred_sig)
            #exp_sig = tf.exp(pred_sig) # avoid pred_sig is too small such as zero    
            #precision = 1./exp_sig
            precision = 1./pred_sig
            #log_loss= 0.5*tf.log(exp_sig)+0.5*precision*((pred_u-y_true)**2)
            log_loss= 0.5*tf.log(pred_sig)+0.5*precision*((pred_u-y_true)**2)            
          
            log_loss=tf.reduce_mean(log_loss)
            return log_loss

        print(self.model.summary())
        self.model.compile(optimizer = self.optimizer, loss=_mve_loss)

    def sample_batch(self, data_inputs, ground_truth, ruitu_inputs, batch_size, certain_id=None, certain_feature=None):
        max_i, _, max_j, _ = data_inputs.shape # Example: (1148, 37, 10, 9)-(sample_ind, timestep, sta_id, features)

        id_ = np.random.randint(max_j, size=batch_size)
        i = np.random.randint(max_i, size=batch_size)
        batch_inputs = data_inputs[i,:,id_,:]
        batch_ouputs = ground_truth[i,:,id_,:]
        batch_ruitu = ruitu_inputs[i,:,id_,:]
        # id used for embedding
        if self.id_embd and (not self.time_embd): 
            expd_id = np.expand_dims(id_,axis=1)
            batch_ids = np.tile(expd_id,(1,37))
            return batch_inputs, batch_ruitu, batch_ouputs, batch_ids
        elif (not self.id_embd) and (self.time_embd):
            time_range = np.array(range(37))
            batch_time = np.tile(time_range,(batch_size,1))
            #batch_time = np.expand_dims(batch_time, axis=-1)

            return batch_inputs, batch_ruitu, batch_ouputs, batch_time
        elif (self.id_embd) and (self.time_embd):
            expd_id = np.expand_dims(id_,axis=1)
            batch_ids = np.tile(expd_id,(1,37))

            time_range = np.array(range(37))
            batch_time = np.tile(time_range,(batch_size,1))
            #batch_time = np.expand_dims(batch_time, axis=-1)

            return batch_inputs, batch_ruitu, batch_ouputs, batch_ids, batch_time
        
        elif (not self.id_embd) and (not self.time_embd): 
            return batch_inputs, batch_ruitu, batch_ouputs

    def fit(self, train_input_obs, train_input_ruitu, train_labels,
            val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, batch_size,
            iterations=300, validation=True):
                
        print('Train batch size: {}'.format(batch_size))
        print('Validation on data size of {};'.format(val_input_obs.shape[0]))
        
        early_stop_count = 0

        for i in range(iterations):
            batch_inputs, batch_ruitu, batch_labels, batch_ids, batch_time = self.sample_batch(train_input_obs, train_labels, 
                                                                         train_input_ruitu, batch_size=batch_size)
            #batch_placeholders = np.zeros_like(batch_labels)

            loss_ = self.model.train_on_batch(x=[batch_inputs, batch_ruitu, batch_ids, batch_time], 
                  y=[batch_labels])

            if (i+1)%50 == 0:
                print('Iteration:{}/{}. Training batch MLE loss:{}'.
                      format(i+1, iterations, loss_))
                
                if validation :
                    self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, each_station_display=False)
                    if len(self.val_loss_list) >0: # Early stopping
                        if(self.current_mean_val_loss) <= min(self.val_loss_list): # compare with the last early_stop_limit values except SELF
                            early_stop_count = 0
                            model_json = self.model.to_json()
                            with open(self.model_save_path+self.model_structure_name, "w") as json_file:
                                json_file.write(model_json)

                            self.model.save_weights(self.model_save_path+self.model_weights_name)
                        else:
                            early_stop_count +=1
                            print('Early-stop counter:', early_stop_count)
                    if early_stop_count == self.early_stop_limit:
                        self.EARLY_STOP=True                    
                        break
        
        print('###'*10)
        if self.EARLY_STOP:
            print('Loading the best model before early-stop ...')
            self.model.load_weights(self.model_save_path+self.model_weights_name)

        print('Training finished! Detailed val MLE loss:')
        self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, each_station_display=True)

    def evaluate(self, data_input_obs, data_input_ruitu, data_labels, data_ids, data_time, each_station_display=False):     
        all_loss=[]
        for i in range(10): # iterate for each station. (sample_ind, timestep, staionID, features)
            #batch_placeholders = np.zeros_like(data_labels[:,:,i,:])
            val_loss= self.model.evaluate(x=[data_input_obs[:,:,i,:], data_input_ruitu[:,:,i,:], data_ids[:,:,i], data_time],
                                y=[data_labels[:,:,i,:]], verbose=False)

            all_loss.append(val_loss)

            if each_station_display:
                print('\tFor station 9000{}, val MLE loss: {}'.format(i+1, val_loss))
        
        self.current_mean_val_loss = np.mean(all_loss)
        print('Mean val MLE loss:', self.current_mean_val_loss)

        self.val_loss_list.append(self.current_mean_val_loss)

    def predict(self, batch_inputs, batch_ruitu, batch_ids, batch_times):
        '''
        Input:

        Output:
        pred_result (mean value) : (None, 10,37,3). i.e., (sample_nums, stationID, timestep, features)
        pred_var_result (var value) : (None, 10,37,3)

        '''
        pred_result_list = []
        pred_var_list = []
        #pred_std_list =[]

        for i in range(10):
            result = self.model.predict(x=[batch_inputs[:,:,i,:], batch_ruitu[:,:,i,:], batch_ids[:,:,i], batch_times])
            var_result = result[:,:,3:6] # Variance
            result = result[:,:,0:3] # Mean

            #result = np.squeeze(result, axis=0)
            pred_result_list.append(result)

            #var_result = np.squeeze(var_result, axis=0)
            pred_var_list.append(var_result)

        pred_result = np.stack(pred_result_list, axis=1)
        pred_var_result = np.stack(pred_var_list, axis=1)

        print('Predictive shape (None, 10,37,3) means (sample_nums, stationID, timestep, features). \
            Features include: t2m, rh2m and w10m')
        self.pred_result = pred_result
        self.pred_var_result = pred_var_result
        #self.pred_std_result = np.sqrt(np.exp(self.pred_var_result[:,:,i,j])) # Calculate standard deviation

        return pred_result, pred_var_result

    def renorm_for_visualization(self, obs_inputs, ruitu_inputs, pred_mean_result, pred_var_result, ground_truth=None):
        '''
        obs_inputs: (None, 28, 10, 9)
        ruitu_inputs: (None, 37, 10, 29)
        pred_mean_result: (None, 10, 37, 3)
        pred_var_result: (None, 10, 37, 3)
        ground_truth: (None, 37, 10, 3)
                
        #self.target_list=['t2m','rh2m','w10m']
        #self.obs_range_dic={'t2m':[-30,42],
        #                 'rh2m':[0.0,100.0],
        #                 'w10m':[0.0, 30.0]}

        #self.obs_and_output_feature_index_map = {'t2m':0,'rh2m':1,'w10m':2}
        #self.ruitu_feature_index_map = {'t2m':1,'rh2m':3,'w10m':4}
        
        #TODO:
        '''
        for target_v in self.target_list:
            temp1 = obs_inputs[:,:,:,self.obs_and_output_feature_index_map[target_v]]
            temp2 = ruitu_inputs[:,:,:,self.ruitu_feature_index_map[target_v]]
            temp3 = pred_mean_result[:,:,:,self.obs_and_output_feature_index_map[target_v]]
            #temp4 = pred_var_result[:,:,:,self.obs_and_output_feature_index_map[target_v]]

            
            obs_inputs[:,:,:,self.obs_and_output_feature_index_map[target_v]] = renorm(temp1, self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
            ruitu_inputs[:,:,:,self.ruitu_feature_index_map[target_v]] = renorm(temp2, self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
            pred_mean_result[:,:,:,self.obs_and_output_feature_index_map[target_v]] = renorm(temp3, self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
            #pred_var_result[:,:,:,self.obs_and_output_feature_index_map[target_v]] = renorm(temp4, self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])

            if ground_truth is not None:
                temp5 = ground_truth[:,:,:,self.obs_and_output_feature_index_map[target_v]]
                ground_truth[:,:,:,self.obs_and_output_feature_index_map[target_v]] = renorm(temp5, self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
        
        if ground_truth is not None:
            return obs_inputs, ruitu_inputs, pred_mean_result, pred_var_result, ground_truth
        else:
            return obs_inputs, ruitu_inputs, pred_mean_result, pred_var_result

    def calc_uncertainty_info(self, verbose=False):
        '''
        Verbose: Display uncertainty for each feature i.e., (t2m, rh2m, w10m) 
        #TODO: Refactor the double 'for' part. 

        '''
        assert len(self.pred_var_result)>0, 'Error! You must run predict() before running calc_uncertainty_info()'
        print('The uncertainty info are calculated on {} predicted samples with shape {}'
            .format(len(self.pred_var_result), self.pred_var_result.shape))

        #
        if verbose:
            assert self.target_list  == ['t2m','rh2m','w10m'], 'ERROR, list changed!'
            
            for j, target_v in enumerate(['t2m','rh2m','w10m']):
                print('For feature {}:'.format(target_v))
                for i in range(37):
                    #unctt_var = np.exp(self.pred_var_result[:,:,i,j])
                    unctt_std = np.sqrt(unctt_var)
                    unctt_mean_std = np.mean(unctt_std)
                    unctt_mean_var = np.mean(unctt_var)
                    #renorm_unctt_mean_std = renorm(unctt_mean_std, self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])                  
                    print('\tTime:{}-Variance:{:.4f}; Std:{:.4f};'.
                        format(i+1, unctt_mean_var, unctt_mean_std))
        else:    
            for i in range(37):
                unctt_var = np.exp(self.pred_var_result[:,:,i,:])
                unctt_std = np.sqrt(unctt_var)
                unctt_mean_std = np.mean(unctt_std)
                unctt_mean_var = np.mean(unctt_var)
                #renorm_unctt_mean_std = 0
                print('Time:{}-Variance:{:.4f}; Std:{:.4f};'.
                    format(i+1, unctt_mean_var, unctt_mean_std))

    def minus_plus_std_strategy(self, pred_mean, pred_var, feature_name,\
                            timestep_to_ensemble=21, alpha=0):
        '''
        This stratergy aims to calculate linear weighted at specific timestep (timestep_to_ensemble) between prediction and ruitu as formula:
                                    (alpha)*pred_mean + (1-alpha)*ruitu_inputs
        pred_mean: (10, 37, 3)
        pred_var: (10, 37, 3)
        timestep_to_ensemble: int32 (From 0 to 36)
        '''
        print('Using minus_plus_var_strategy with alpha {}'.format(alpha))
        assert 0<=timestep_to_ensemble<=36 , 'Please ensure 0<=timestep_to_ensemble<=36!'
        assert -0.3<= alpha <=0.3, '-0.3<= alpha <=0.3!'
        assert pred_mean.shape == (10, 37, 3), 'Error! This funtion ONLY works for \
        one data sample with shape (10, 37, 3). Any data shape (None, 10, 37, 3) will leads this error!'
        pred_std = np.sqrt(np.exp(pred_var))           
        print('alpha:',alpha)

        pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] = \
        pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] + \
        alpha * pred_std[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]]

        return pred_mean

    def linear_ensemble_strategy(self, pred_mean, pred_var, ruitu_inputs, feature_name,\
                            timestep_to_ensemble=21, alpha=1):
        '''
        This stratergy aims to calculate linear weighted at specific timestep (timestep_to_ensemble) between prediction and ruitu as formula:
                                    (alpha)*pred_mean + (1-alpha)*ruitu_inputs
        pred_mean: (10, 37, 3)
        pred_var: (10, 37, 3)
        ruitu_inputs: (37,10,29). Need Swamp to(10,37,29) FIRSTLY!!
        timestep_to_ensemble: int32 (From 0 to 36)
        '''
        assert 0<= alpha <=1, 'Please ensure 0<= alpha <=1 !'
        assert pred_mean.shape == (10, 37, 3), 'Error! This funtion ONLY works for \
        one data sample with shape (10, 37, 3). Any data shape (None, 10, 37, 3) will leads this error!'
        #pred_std = np.sqrt(np.exp(pred_var))           
        ruitu_inputs = np.swapaxes(ruitu_inputs,0,1)
        print('alpha:',alpha)

        pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] = \
        (alpha)*pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] + \
                                (1-alpha)*ruitu_inputs[:,timestep_to_ensemble:, self.ruitu_feature_index_map[feature_name]]  
        print('Corrected pred_mean shape:', pred_mean.shape)
        
        return pred_mean

    def fuzzy_ensemble_strategy(self, pred_mean, pred_var, feature_name,\
                            timestep_to_ensemble=21, alpha=0):
        '''
        This stratergy aims to calculate linear weighted at specific timestep (timestep_to_ensemble) between prediction and ruitu as formula:
                                    (alpha)*pred_mean + (1-alpha)*ruitu_inputs
        pred_mean: (10, 37, 3)
        pred_var: (10, 37, 3)
        timestep_to_ensemble: int32 (From 0 to 36)
        '''
        print('Using fuzzy_ensemble_strategy with alpha {}'.format(alpha))

        assert 0<=timestep_to_ensemble<=36 , 'Please ensure 0<=timestep_to_ensemble<=36!'
        assert -0.4<= alpha <=0.4, 'Please ensure -0.4<= alpha <=0.4 !'
        assert pred_mean.shape == (10, 37, 3), 'Error! This funtion ONLY works for \
        one data sample with shape (10, 37, 3). Any data shape (None, 10, 37, 3) will leads this error!'
        
        pred_std = np.sqrt(np.exp(pred_var))           
        #print('normalizing for Std. after timestep:', timestep_to_ensemble)
        temp_std = pred_std[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]]
        norm_std = temp_std / np.max(temp_std)
        #print('norm_std shape', norm_std.shape)
        dim_0, dim_1 = norm_std.shape
        reshaped_std = norm_std.reshape(-1)     
        from skfuzzy import trimf
        fuzzy_degree = trimf(reshaped_std, [0., 1, 1.2])
        fuzzy_degree = fuzzy_degree.reshape(dim_0, dim_1)
        #print('fuzzy_degree shape:',fuzzy_degree.shape)
        #print('temp_std shape:',temp_std.shape)

        pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] = \
        pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] + \
        fuzzy_degree*alpha*temp_std
        #pred_mean[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]] + \
        #alpha * pred_std[:,timestep_to_ensemble:,self.obs_and_output_feature_index_map[feature_name]]
        #print('pred_mean.shape',pred_mean.shape)
        return pred_mean

        pass

    def renorm_for_submit(self, pred_mean, pred_var, ruitu_inputs, timestep_to_ensemble=21, alpha=1):
        ''' 
        Overwrite for Seq2Seq_MVE Class
        pred_mean: shape of (10, 37, 3)
        pred_var: shape of (10, 37, 3)
        ruitu_inputs: shape of (10, 37, 3)
        timestep_to_ensemble: int32 (From 0 to 36)

        # TODO: Add three strategies for output
        '''
        assert self.pred_result is not None, 'You must run self.predict(batch_inputs, batch_ruitu) firstly!!'
        assert pred_mean.shape == (10, 37, 3), 'Error! This funtion ONLY works for one data sample with shape (10, 37, 3). Any data shape (None, 10, 37, 3) will leads this error!'

        df_empty = pd.DataFrame(columns=['FORE_data', 't2m', 'rh2m', 'w10m'])

        for j, target_v in enumerate(self.target_list):
            
            series_ids = pd.Series()
            series_targets = pd.Series()

            #print('Feature {}, timestep_to_ensemble: {}, weighted alpha: {}'.
            #    format(target_v, timestep_to_ensemble, alpha))

            #pred_mean = self.linear_ensemble_strategy(pred_mean, pred_var, 
            #    ruitu_inputs, target_v, timestep_to_ensemble, alpha)

            #pred_mean =self.minus_plus_std_strategy(pred_mean, pred_var, target_v,\
            #                timestep_to_ensemble, alpha)

            #pred_mean = self.fuzzy_ensemble_strategy(pred_mean, pred_var, target_v,\
            #                timestep_to_ensemble, alpha=0.)

            renorm_value = renorm(pred_mean[:,:,j], self.obs_range_dic[target_v][0], self.obs_range_dic[target_v][1])
                
            for i in range(10):
                if i != 9:
                    id_num = '0'+str(i+1)
                else:
                    id_num = str(10)
                sta_name_time = '900'+id_num+'_'
                
                time_str_list=[]

                for t in range(37):
                    if t < 10:
                        time_str= sta_name_time + '0'+ str(t)
                    else:
                        time_str = sta_name_time + str(t)
                    
                    time_str_list.append(time_str)
                
                series_id = pd.Series(time_str_list)
                series_target = pd.Series(renorm_value[i])
                
                series_ids = pd.concat([series_ids, series_id])
                series_targets = pd.concat([series_targets, series_target])
                
            df_empty['FORE_data'] = series_ids
            df_empty[target_v] = series_targets

        return df_empty

    def plot_prediction(self, x, y_true, y_pred, intervals=None, input_ruitu=None, pi_degree=0.8, renorm_flag=False):
        """Plots the predictions.

        Arguments
        ---------
        x: Input sequence of shape (input_sequence_length,
            dimension_of_signal) E.g. (28, 1)
        y_true: True output sequence of shape (input_sequence_length,
            dimension_of_signal) E.g. (35, 1)
        y_pred: Predicted output sequence (input_sequence_length,
            dimension_of_signal) E.g. (35, 1)
        input_ruitu: Ruitu output sequence E.g. (35, 1)
        pi_degree: Confident Level such as 0.95, 0.9, 0.8, and 0.68 etc.
        """

        plt.figure(figsize=(12, 3))


        output_dim = x.shape[-1]# feature dimension
        for j in range(output_dim):
            past = x[:, j] 
            true = y_true[:, j]
            pred = y_pred[:, j]
            if input_ruitu is not None:
                ruitu = input_ruitu[:, j]
            if intervals is not None:
                pi_var = intervals[:, j]
                pi_var = np.sqrt(np.exp(pi_var))
                
            label1 = "Seen (past) values" if j==0 else "_nolegend_"
            label2 = "True future values" if j==0 else "_nolegend_"
            label3 = "Predictions" if j==0 else "_nolegend_"
            label4 = "Ruitu values" if j==0 else "_nolegend_"
            label5 = "Lower-Upper bound" if j==0 else "_nolegend_"

            plt.plot(range(len(past)), past, "o-g",
                     label=label1)
            plt.plot(range(len(past),
                     len(true)+len(past)), true, "x--g", label=label2)
            plt.plot(range(len(past), len(pred)+len(past)), pred, ".--b",
                     label=label3)
            if input_ruitu is not None:
                plt.plot(range(len(past), len(ruitu)+len(past)), ruitu, ".--r",
                     label=label4)
            if intervals is not None:
                #print(intervals.shape)
                print(pi_var.shape)
                up_bound = pred + self.pi_dic[pi_degree]*pi_var
                low_bound = pred - self.pi_dic[pi_degree]*pi_var
                plt.fill_between(range(len(past), len(ruitu)+len(past)), 
                                 up_bound, low_bound, facecolor='blue', alpha=0.1)              
                
        plt.legend(loc='best')
        plt.title("Predictions v.s. true values v.s. Ruitu")
        plt.show()

class Enc_Dec_Embd(Enc_Dec):
    def build_graph(self):
        keras.backend.clear_session() # clear session/graph    
        self.optimiser = keras.optimizers.Adam(lr=self.lr, decay=self.decay)
        # Define an input sequence.
        encoder_inputs = keras.layers.Input(shape=(None, self.num_input_features), name='encoder_inputs')
        # Create a list of RNN Cells, these are then concatenated into a single layer
        # with the RNN layer.
        encoder_cells = []
        for hidden_neurons in self.layers:
            encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                      kernel_regularizer = self.regulariser,
                                                      recurrent_regularizer = self.regulariser,
                                                      bias_regularizer = self.regulariser))
            
        encoder = keras.layers.RNN(encoder_cells, return_state=True)
        encoder_outputs_and_states = encoder(encoder_inputs)
        # Discard encoder outputs and only keep the states.
        encoder_states = encoder_outputs_and_states[1:]
        # Define a decoder sequence.
        decoder_inputs = keras.layers.Input(shape=(None, self.num_decoder_features), name='decoder_inputs')
        
        decoder_inputs_id = keras.layers.Input(shape=(None,), name='id_inputs')
        decoder_inputs_id_embd = Embedding(input_dim=10, output_dim=2, name='id_embedding')(decoder_inputs_id)

        #decoder_inputs_time = keras.layers.Input(shape=(None,), name='time_inputs')
        #decoder_inputs_time_embd = Embedding(input_dim=37, output_dim=2, name='time_embedding')(decoder_inputs_time)

        decoder_concat = concatenate([decoder_inputs, decoder_inputs_id_embd], axis=-1)

        decoder_cells = []
        for hidden_neurons in self.layers:
            decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                      kernel_regularizer = self.regulariser,
                                                      recurrent_regularizer = self.regulariser,
                                                      bias_regularizer = self.regulariser))

        decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
        decoder_outputs_and_states = decoder(decoder_concat, initial_state=encoder_states)

        decoder_outputs = decoder_outputs_and_states[0]

        #decoder_dense1 = keras.layers.Dense(units=32,
        #                                   activation='relu',
        #                                   kernel_regularizer = self.regulariser,
        #                                  bias_regularizer = self.regulariser, name='dense_relu')

        output_dense = keras.layers.Dense(self.num_output_features,
                                           activation='sigmoid',
                                           kernel_regularizer = self.regulariser,
                                           bias_regularizer = self.regulariser, name='output_sig')

        #densen1=decoder_dense1(decoder_outputs)
        decoder_outputs = output_dense(decoder_outputs)
        # Create a model using the functional API provided by Keras.
        self.model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_id], outputs=decoder_outputs)
        self.model.compile(optimizer = self.optimiser, loss=self.loss)

        print(self.model.summary())
    
    def sample_batch(self, data_inputs, ground_truth, ruitu_inputs, batch_size, certain_id=None, certain_feature=None):
        
        max_i, _, max_j, _ = data_inputs.shape # Example: (1148, 37, 10, 9)-(sample_ind, timestep, sta_id, features)
        
        if certain_id == None and certain_feature == None:
            id_ = np.random.randint(max_j, size=batch_size)
            i = np.random.randint(max_i, size=batch_size)
            batch_inputs = data_inputs[i,:,id_,:]
            batch_ouputs = ground_truth[i,:,id_,:]
            batch_ruitu = ruitu_inputs[i,:,id_,:]

            # id used for embedding
            expd_id = np.expand_dims(id_,axis=1)
            batch_ids = np.tile(expd_id,(1,37))
            #batch_time = 

        elif certain_id != None:
            pass

        return batch_inputs, batch_ruitu, batch_ouputs, batch_ids

    def fit(self, train_input_obs, train_input_ruitu, train_labels,
            val_input_obs, val_input_ruitu, val_labels, val_ids, batch_size,
            iterations=300, validation=True):
                
        print('Train batch size: {}'.format(batch_size))
        print('Validation on data size of {};'.format(val_input_obs.shape[0]))

        for i in range(iterations):
            batch_inputs, batch_ruitu, batch_labels, batch_ids = self.sample_batch(train_input_obs, train_labels, 
                                                                         train_input_ruitu, batch_size=batch_size)
            #print(batch_ids.shape)
            loss_ = self.model.train_on_batch(x=[batch_inputs, batch_ruitu, batch_ids], 
                  y=[batch_labels])

            if (i+1)%50 == 0:
                print('Iteration:{}/{}. Training batch loss:{}'.
                      format(i+1, iterations, loss_))
                if validation :
                    self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, each_station_display=False)                
        
        print('###'*10)
        print('Train finish! Total validation loss:')

        self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, each_station_display=True)

    def evaluate(self, data_input_obs, data_input_ruitu, data_labels, data_ids, each_station_display=False, each_feature_display=False):
        #assert data_input_ruitu.shape[0] == data_input_obs.shape[0] == data_labels.shape[0], 'Shape Error'
        #assert data_input_obs.shape[1] == 28 and data_input_obs.shape[2] == 10 and data_input_obs.shape[3] == 9, 'Error! Obs input shape must be (None, 28,10,9)'
        #assert data_input_ruitu.shape[1] == 37 and data_input_ruitu.shape[2] == 10 and data_input_ruitu.shape[3] == 29, 'Error! Ruitu input shape must be (None, 37,10,29)'
        #assert data_labels.shape[1] == 37 and data_labels.shape[2] == 10 and data_labels.shape[3] == 3, 'Error! Ruitu input shape must be (None, 37,10,3)' 
        
        all_loss=[]
        for i in range(10): # iterate for each station. (sample_ind, timestep, staionID, features)
            #print(data_ids.shape)
            val_loss= self.model.evaluate(x=[data_input_obs[:,:,i,:], data_input_ruitu[:,:,i,:], data_ids[:,:,i]],
                                y=data_labels[:,:,i,:], verbose=False)

            all_loss.append(val_loss)

            if each_station_display:
                print('\tFor station 9000{}, evaluated loss: {}'.format(i+1, val_loss))
        
        print('Mean evaluated loss on all stations:', np.mean(all_loss))

    def predict(self, batch_inputs, batch_ruitu, batch_ids):
        #assert batch_ruitu.shape[0] == batch_inputs.shape[0], 'Shape Error'
        #assert batch_inputs.shape[1] == 28 and batch_inputs.shape[2] == 10 and batch_inputs.shape[3] == 9, 'Error! Obs input shape must be (None, 28,10,9)'
        #assert batch_ruitu.shape[1] == 37 and batch_ruitu.shape[2] == 10 and batch_ruitu.shape[3] == 29, 'Error! Ruitu input shape must be (None, 37,10, 29)'
        pred_result_list = []
        for i in range(10):
            #print('Predict for station: 9000{}'.format(i+1))
            result = self.model.predict(x=[batch_inputs[:,:,i,:], batch_ruitu[:,:,i,:], batch_ids[:,:,i]])
            result = np.squeeze(result, axis=0)
            #all_pred[i] = result
            pred_result_list.append(result)
            #pass

        pred_result = np.stack(pred_result_list, axis=0)
        #return all_pred, pred_result
        print('Predict shape (10,37,3) means (stationID, timestep, features). Features include: t2m, rh2m and w10m')
        self.pred_result = pred_result
        return pred_result

def renorm(norm_value, min_v ,max_v):
    real_v = norm_value * (max_v-min_v) + min_v
    return real_v