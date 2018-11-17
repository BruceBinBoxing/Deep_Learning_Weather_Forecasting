import numpy as np
np.random.seed(123)  # for reproducibility
import pandas as pd
import requests
import os
import pickle as pk
import configparser
import datetime
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def pred_batch_iter(sourceData, labelData, batch_size, num_epochs, shuffle=False):
	data = np.array(sourceData)  # 将sourceData转换为array存储
	data_size = len(sourceData)
	num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			os.__exit()
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = sourceData[shuffle_indices]           
		else:
		    input_data = sourceData
		    label_data = labelData

		for batch_num in range(num_batches_per_epoch):
		    start_index = batch_num * batch_size
		    end_index = min((batch_num + 1) * batch_size, data_size)
		    if (end_index-start_index) < batch_size: # Padding! In case that the last size of batch < batch_size
		        start_index = end_index-batch_size
		    
		    yield input_data[start_index:end_index], label_data[start_index:end_index]

def get_random_batch(sourceData, labelData, batch_size):
	data_size = len(sourceData)
	shuffle_indices = np.random.permutation(np.arange(data_size))
	shuffled_data = sourceData[shuffle_indices]
	shuffled_labels = labelData[shuffle_indices]

	return shuffled_data[:batch_size], shuffled_labels[:batch_size]

def batch_iter(sourceData, batch_size, num_epochs, shuffle=False):
    data = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if (end_index-start_index) < batch_size: # Padding! In case that the last size of batch < batch_size
                start_index = end_index-batch_size
            yield shuffled_data[start_index:end_index]

def save_pkl(pkl_file, file_path, file_name, min_v=None, max_v=None):
    pk.dump(pkl_file, open(os.path.join(file_path, file_name), "wb"))
    print(file_name,' is dumped in: ', file_path)
    if((min_v is not None) and (max_v is not None)):
        pk.dump(min_v, open(os.path.join(file_path,file_name+"min"), "wb"))
        print(file_name+".min",' is dumped in: ', file_path)
        pk.dump(max_v, open(os.path.join(file_path,file_name+"max"), "wb"))
        print(file_name+".max",' is dumped in: ', file_path)

def load_pkl(file_path, file_name):
    pkl_file = pk.load(open(os.path.join(file_path,file_name), "rb"))
    print(file_name,' is loaded from: ', file_path)  
    return pkl_file

def split_data(data, labels, ratio_=0.8, shuffle=False):
	if shuffle:
		data_size = len(data)
		shuffle_indices = np.random.permutation(np.arange(data_size))
		data = data[shuffle_indices]
		labels = labels[shuffle_indices]

	train_X = data[:int(len(data) * ratio_)]
	test_X = data[int(len(data) * ratio_):]

	train_Y = labels[:int(len(labels) * ratio_)]
	test_Y = labels[int(len(labels) * ratio_):]

	return train_X, train_Y, test_X, test_Y

def get_ndarray_by_sliding_window(data_df, input_len, output_len, vars_list, only_target=False):
    i=0
    X=[]
    Y=[]  
    while True:
        if (i+input_len+output_len) <= len(data_df):
            X.append(data_df[i:i+input_len][vars_list].values)
            Y.append(data_df[i+input_len:i+input_len+output_len][vars_list].values)
            i+=1
        else:
            X=np.array(X)
            Y=np.array(Y)
            assert len(Y) == len(X), 'Length Error !!!!!!!'
            break            
    return X, Y
    
def get_train_test(data_df, input_len, output_len, var_name, per=0.9, only_target=True, data_name='obs'):
    i=0
    X=[]
    Y=[]    
    #if data_name=='obs':        
    #    targets = ['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
        
    #elif data_name=='ruitu':
        
    #    targets = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M',
    #   'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M',
    #   'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M',
    #   'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M',
    #   'Q850_M', 'Q700_M', 'Q500_M']    
    targets = var_name[data_name]
    
    while True:
        if (i+input_len+output_len) <= len(data_df):
            X.append(data_df[i:i+input_len][targets].values)
            Y.append(data_df[i+input_len:i+input_len+output_len][targets].values)
            i+=1
        else:
            X=np.array(X)
            Y=np.array(Y)
            assert len(Y) == len(X), 'Length Error'
            break
            
    train_X, train_Y, test_X, test_Y = split_data(X, Y, ratio_=per)
    
    if data_name=='ruitu':
        return train_X, train_Y, test_X, test_Y
    
    elif (data_name=='obs' and only_target):
        # Only return the first three variables, i.e., ['t2m_obs','rh2m_obs','w10m_obs']
        return train_X[:,:,:3], train_Y[:,:,:3], test_X[:,:,:3], test_Y[:,:,:3] 
    
    else:
        return train_X, train_Y[:,:,:3], test_X, test_Y[:,:,:3]

def cal_loss_dataset(pred_batch_iter, X, Y, batch_num,  ):
    get_next_batch = pred_batch_iter(X, Y, batch_size=batch_num,
                                     num_epochs=1,shuffle=False) # batch iterator    
    loss_list=[]
    for X_batch, Y_batch in (get_next_batch):

        X_batch = X_batch.reshape([-1, step_num_in, elem_num])
        X_batch = Y_batch.reshape([-1, step_num_out, elem_num])

        (loss_valid_batch) = sess.run([seq2seq_train.loss],
                           {p_input: X_valid_batch, 
                            p_label:Y_valid_batch})

        loss_valid_list.append(loss_valid_batch)

    loss_avg_validset = np.mean(np.array(loss_valid_list), dtype=np.float32)
    
    return loss_avg_validset

def rmse(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    #print(y_true.shape)
    return np.sqrt(mean_squared_error(y_pred, y_true))
def mse(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return mean_squared_error(y_pred, y_true)
def bias(y_pred, y_true):
    pass
def mae(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return np.mean(np.abs(y_pred-y_true))
def score(y_pred, y_true):
    pass

def evl_fn(y_pred, y_true, **kwargs):
    renorm = kwargs['renorm']
    min_v = kwargs['max_min'][0]
    max_v = kwargs['max_min'][1]
    
    if renorm:
        y_pred = y_pred * (max_v-min_v) + min_v
        y_true = y_true * (max_v-min_v) + min_v
        
        print('\t rmse:', rmse(y_pred, y_true) )
        print('\t mae: ', mae(y_pred, y_true) )
        print('\t mse: ', mse(y_pred, y_true) )
    else:
        print('\t rmse:', rmse(y_pred, y_true))
        print('\t mae: ', mae(y_pred, y_true))
        print('\t mse: ', mse(y_pred, y_true))
#print('Baseline_direct: rmse:{}, mse:{}'.format(rmse(X,Y), mse(X, Y)))

def renorm(norm_value, min_v ,max_v):
    real_v = norm_value * (max_v-min_v) + min_v
    return real_v
    

def cal_miss(X_miss):
    nums_ = len(X_miss.reshape(-1))
    miss_nums = np.sum(X_miss == -9999)
    print('all nums:', nums_)
    print('missing nums:', miss_nums)
    print('missing ratio:', miss_nums/nums_)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def intplt_nan_1d(data_nan, obs_data_nan, sta_id): # Default stationID
    '''
    data_nan: is Ruitu data with np.NaN
    sta_id: Is only one stationID;
    obs_data_nan: is Observation data with np.NaN
    
    '''
    data_nan[data_nan == -9999] = np.NaN
    obs_data_nan[obs_data_nan == -9999] = np.NaN
    
    data_nan=data_nan[:,:,sta_id]
    obs_data_nan=obs_data_nan[:,:,sta_id]

    new_list=[]
    print('Original Ruitu Data Shape:', data_nan.shape)
    print('Original Observed Data Shape:', obs_data_nan.shape)
    
    #print('Firstly, we delete the totally lost days in Obs dataset and the counterpart day in Ruitu dataset.')
    day_should_deleted=[]
    for i in range(obs_data_nan.shape[0]):
        if np.isnan(obs_data_nan[i,:]).any():
            if sum(np.isnan(obs_data_nan[i,:])) == 37:                
                day_should_deleted.append(i)
                continue    
    print('Data are totally lost during the days in obs dataset!', day_should_deleted)
    obs_data_nan = np.array(np.delete(obs_data_nan, day_should_deleted, 0))
    data_nan = np.array(np.delete(data_nan, day_should_deleted, 0))    
    #---------------------------------------------------------
    
    #print('Secondly, we delete the totally lost days in Ruitu dataset and the counterpart day in Obs dataset.')
    day_should_deleted=[]
    for i in range(data_nan.shape[0]):
        if np.isnan(data_nan[i,:]).any():
            if sum(np.isnan(data_nan[i,:])) == 37:
                day_should_deleted.append(i)
                continue
    print('Data are totally lost during the days in Ruitu dataset!', day_should_deleted)
    obs_data_nan = np.array(np.delete(obs_data_nan, day_should_deleted, 0))
    data_nan = np.array(np.delete(data_nan, day_should_deleted, 0))    
    #---------------------------------------------------------
    
    ### Interpolate for Input data
    for i in range(data_nan.shape[0]):
        #print(i)
        new_X = data_nan[i,:].copy()
        if np.isnan(new_X).any():
            nans, x_temp= nan_helper(new_X)
            new_X[nans]= np.interp(x_temp(nans), x_temp(~nans), new_X[~nans])                    
        new_list.append(new_X)
    data_after_intplt = np.array(new_list)
        
    ###Interpolate for Label(Obs) data
    Y_list=[]
    for i in range(obs_data_nan.shape[0]):
        new_Y = obs_data_nan[i,:].copy()
        if np.isnan(new_Y).any():
            #print('Miss happen! Interpolate...')
            nans, y_temp= nan_helper(new_Y)
            #print(np.isnan(new_Y))
            new_Y[nans]= np.interp(y_temp(nans), y_temp(~nans), new_Y[~nans])                    
        Y_list.append(new_Y)
    
    obs_after_intplt =  np.array(Y_list)
    print('After interpolate, Ruitu Data Shape:', data_after_intplt.shape)
    print('After interpolate, Observed Data Shape:', obs_after_intplt.shape)
    return data_after_intplt, obs_after_intplt

def min_max_norm(org_data,min_,max_):
    return (org_data-min_)/(max_-min_)

def renorm_for_submit(pred_mean, pred_var, ruitu_inputs, timestep_to_ensemble=21, alpha=1):

        ''' 
        Overwrite for Seq2Seq_MVE Class
        pred_mean: shape of (10, 37, 3)
        pred_var: shape of (10, 37, 3)
        ruitu_inputs: shape of (10, 37, 3)
        timestep_to_ensemble: int32 (From 0 to 36)

        # TODO: Add three strategies for output
        '''
        obs_range_dic={'t2m':[-30,42], # Official value: [-20,42]
                         'rh2m':[0.0,100.0],
                         'w10m':[0.0, 30.0]}
        assert pred_mean.shape == (10, 37, 3), 'Error! This funtion ONLY works for one data sample with shape (10, 37, 3). Any data shape (None, 10, 37, 3) will leads this error!'

        df_empty = pd.DataFrame(columns=['FORE_data', 't2m', 'rh2m', 'w10m'])

        for j, target_v in enumerate(['t2m','rh2m','w10m']):
            
            series_ids = pd.Series()
            series_targets = pd.Series()
            renorm_value = renorm(pred_mean[:,:,j], obs_range_dic[target_v][0], obs_range_dic[target_v][1])
                
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

def predict(model, batch_inputs, batch_ruitu, batch_ids, batch_times):
        pred_result_list = []
        pred_var_list = []
        for i in range(10):
            result = model.predict(x=[batch_inputs[:,:,i,:], batch_ruitu[:,:,i,:], batch_ids[:,:,i], batch_times])
            var_result = result[:,:,3:6] # Variance
            result = result[:,:,0:3] # Mean
            pred_result_list.append(result)
            pred_var_list.append(var_result)

        pred_result = np.stack(pred_result_list, axis=1)
        pred_var_result = np.stack(pred_var_list, axis=1)  
        pred_std = np.sqrt(pred_var_result)
        return pred_result, pred_std