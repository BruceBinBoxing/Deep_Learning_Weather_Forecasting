import numpy as np
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Lambda
import os, sys
from weather_model import Seq2Seq_MVE_subnets_swish, Seq2Seq_MVE, Seq2Seq_MVE_subnets
from keras.models import load_model, model_from_json
from parameter_config_class import parameter_config

model_save_path = '../models/'

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

class Seq2Seq_Class(parameter_config):
    def __init__(self, model_save_path='../models', 
              model_structure_name='seq2seq_model_demo', 
              model_weights_name='seq2seq_model_demo', 
              model_name=None):
        super().__init__()

        self.model_save_path = model_save_path
        self.model_structure_name=model_structure_name + self.model_name_format_str +'.json'
        self.model_weights_name=model_weights_name + self.model_name_format_str +'.h5'
        print('model_structure_name:', self.model_structure_name)
        print('model_weights_name:', self.model_weights_name)

        self.pred_result = None # Predicted mean value
        self.pred_var_result = None # Predicted variance value
        self.current_mean_val_loss = None
        self.EARLY_STOP=False
        self.val_loss_list=[]
        self.train_loss_list=[]
        self.pred_var_result = []

    def build_graph(self):
        #keras.backend.clear_session() # clear session/graph    
        self.optimizer = keras.optimizers.Adam(lr=self.lr, decay=self.decay)

        self.model = Seq2Seq_MVE(id_embd=self.id_embd, time_embd=self.time_embd,
            lr=self.lr, decay=self.decay, 
            num_input_features=self.num_input_features, num_output_features=self.num_output_features,
            num_decoder_features=self.num_decoder_features, layers=self.layers,
            loss=self.loss, regulariser=self.regulariser, dropout_rate = self.dropout_rate)

        def loss_fn(y_true, y_pred):
            pred_u = crop(2,0,3)(y_pred) # mean of Gaussian distribution
            pred_sig = crop(2,3,6)(y_pred) # variance of Gaussian distribution
            if self.loss == 'mve':
                precision = 1./pred_sig
                log_loss= 0.5*tf.log(pred_sig)+0.5*precision*((pred_u-y_true)**2)                                 
                log_loss=tf.reduce_mean(log_loss)
                return log_loss
            elif self.loss == 'mse':
                mse_loss = tf.reduce_mean((pred_u-y_true)**2)
                return mse_loss
            elif self.loss == 'mae':
                mae_loss = tf.reduce_mean(tf.abs(y_true-pred_u))
                return mae_loss
            else:
                sys.exit("'Loss type wrong! They can only be mae, mse or mve'")
                
        print(self.model.summary())
        self.model.compile(optimizer = self.optimizer, loss=loss_fn)

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

        model_json = self.model.to_json()
        with open(self.model_save_path+self.model_structure_name, "w") as json_file:
            json_file.write(model_json)
        print('Model structure has been saved at:', self.model_save_path+self.model_structure_name)

        for i in range(iterations):
            batch_inputs, batch_ruitu, batch_labels, batch_ids, batch_time = self.sample_batch(train_input_obs, train_labels, 
                                                                         train_input_ruitu, batch_size=batch_size)

            loss_ = self.model.train_on_batch(x=[batch_inputs, batch_ruitu, batch_ids, batch_time], 
                  y=[batch_labels])

            if (i+1)%50 == 0:
                print('Iteration:{}/{}. Training batch loss:{}'.
                      format(i+1, iterations, loss_))
                
                if validation :
                    self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, each_station_display=False)
                    if len(self.val_loss_list) >0: # Early stopping
                        if(self.current_mean_val_loss) <= min(self.val_loss_list): # compare with the last early_stop_limit values except SELF
                            early_stop_count = 0
                            self.model.save_weights(self.model_save_path+self.model_weights_name)
                            print('The newest optimal model weights are updated at:', self.model_save_path+self.model_weights_name)
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

        print('Training finished! Detailed val loss with the best model weights:')
        self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, each_station_display=True)

    def evaluate(self, data_input_obs, data_input_ruitu, data_labels, data_ids, data_time, each_station_display=False):     
        all_loss=[]
        for i in range(10): # iterate for each station. (sample_ind, timestep, staionID, features)
            #batch_placeholders = np.zeros_like(data_labels[:,:,i,:])
            val_loss= self.model.evaluate(x=[data_input_obs[:,:,i,:], data_input_ruitu[:,:,i,:], data_ids[:,:,i], data_time],
                                y=[data_labels[:,:,i,:]], verbose=False)

            all_loss.append(val_loss)

            if each_station_display:
                print('\tFor station 9000{}, val loss: {}'.format(i+1, val_loss))
        
        self.current_mean_val_loss = np.mean(all_loss)
        print('Mean val loss:', self.current_mean_val_loss)

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