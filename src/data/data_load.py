import pandas as pd
from datetime import datetime, timedelta
from helper import get_ndarray_by_sliding_window, load_pkl, min_max_norm
import numpy as np

interim_path = '../data/interim/'
processed_path = '../data/processed/'

ruitu_range_dic={'psfc_M':[850,1100],
                't2m_M':[-30,42], # Official value: [-20,42]
                'q2m_M':[-0,30],
                 'rh2m_M':[0.0,100.0],
                 'w10m_M':[0.0, 30.0],
                 'd10m_M':[0.0, 360.0],
                 'u10m_M':[-25.0, 20.0], # Official value: [-20,20]
                 'v10m_M':[-20.0, 20.0],
                 'SWD_M':[0.0, 1200.0],
                 'GLW_M':[0.0, 550.0],
                 'HFX_M':[-200.0, 500.0],
                 'LH_M':[-50.0, 300.0],
                 'RAIN_M':[0.0, 300.0],
                 'PBLH_M':[0.0, 5200.0],
                 'TC975_M':[-30.0, 40.0],
                 'TC925_M':[-35.0, 38.0],
                 'TC850_M':[-38.0, 35.0],
                 'TC700_M':[-45.0, 30.0],
                 'TC500_M':[-70.0, 28.0],
                 'wspd975_M':[0.0, 50.0],
                 'wspd925_M':[0.0, 50.0],
                 'wspd850_M':[0.0, 50.0],
                 'wspd700_M':[0.0, 50.0],
                 'wspd500_M':[0.0, 50.0],
                 'Q975_M':[0.0, 10.0],
                 'Q925_M':[0.0, 10.0],
                 'Q850_M':[0.0, 10.0],
                 'Q700_M':[0.0, 10.0],
                 'Q500_M':[0.0, 5.0],
                }
obs_range_dic={'psur_obs':[850,1100],
                't2m_obs':[-30,42], # Official value: [-20,42]
                'q2m_obs':[0,30],
                 'rh2m_obs':[0.0,100.0],
                 'w10m_obs':[0.0, 30.0],
                 'd10m_obs':[0.0, 360.0],
                 'u10m_obs':[-25.0, 20.0], # Official value: [-20,20]
                 'v10m_obs':[-20.0, 20.0],
                 'RAIN_obs':[0.0, 300.0],}

obs_var=['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

ruitu_var=['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M',
       'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M',
       'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M',
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M',
       'Q850_M', 'Q700_M', 'Q500_M']

vars_names = {'obs':obs_var,
             'ruitu':ruitu_var}

def reset_value_range(df, range_dic):
    '''
    Set outlier value into the normal range according to ruitu_range_dic AND obs_range_dic
    '''
    cols = df.columns
    for c in cols:
        min_ = range_dic[c][0]
        max_ = range_dic[c][1]
        if df[c].min() < min_:
            print('{} min Error! Min should >= {}, but found {}'.format(c, min_, df[c].min()))
            df[c].loc[df[c] < min_] = min_
        if df[c].max() > max_:
            print('{} max Error! Max should <= {}, but found {}'.format(c, max_, df[c].max()))
            df[c].loc[df[c] > max_] = max_
            #assert df[c].max() <= max_, '{} max Error! Max should <= {}, but found {}'.format(c, max_, df[c].max())
    return df
    
def transform_from_df2ndarray(obs_df, ruitu_df, station_id, 
    input_len=74, output_len=37, obs_input_only_target_vars=False):
    # Load imputed Dataframe    
    imputed_obs_df = obs_df
    imputed_ruitu_df = ruitu_df

    assert len(imputed_obs_df.columns) == len(obs_range_dic), 'Error! Length error'
    assert len(imputed_ruitu_df.columns) == len(ruitu_range_dic), 'Error! Length error'
    # non-NaN sanity check
    
    # Transform
    targets=['t2m', 'rh2m', 'w10m']
    
    imputed_ruitu_df.reset_index(inplace=True)
    imputed_obs_df.reset_index(inplace=True)
    
    imputed_ruitu_df.set_index(['sta_id', 'time_index'], inplace=True)
    imputed_obs_df.set_index(['sta_id', 'time_index'], inplace=True)

    #time_format_str='%Y-%m-%d %H:%M:%S'
    #start_time = '2015-03-01 03:00:00'
    #start_date = datetime.strptime(start_time, time_format_str)
    #all_hours = 28512 # TODO: This number should be adaptive according to train, test, validation dataset
    if station_id == 999:
        station_list = [90001, 90002,90003,90004,90005,90006,90007,90008,90009,90010]
    else:
        station_list = [sta_id]    
    #sta_id = station_id

    obs_inputs_=[]
    ruitu_inputs_=[]
    obs_outputs_=[]

    for sta_id in station_list:
        print('Making ndarray for station ID:', sta_id)
        #print(imputed_obs_df.head())
        selected_df_obs = imputed_obs_df.loc[sta_id]
        selected_df_ruitu = imputed_ruitu_df.loc[sta_id]
        
        selected_df_obs = reset_value_range(selected_df_obs, obs_range_dic)
        selected_df_ruitu = reset_value_range(selected_df_ruitu, ruitu_range_dic)
        
        # Max-min normalization
        # normalize for each column
        cols = selected_df_obs.columns
        norm_obs_df = selected_df_obs.copy()
        for c in cols:
            print('Normalizing column {}...'.format(c))
            norm_obs_df[c] = min_max_norm(selected_df_obs[c], obs_range_dic[c][0], obs_range_dic[c][1])

        print('OK! Has normalized for Observation dataframe!')

        # normalize for each column
        cols = selected_df_ruitu.columns
        norm_ruitu_df = selected_df_ruitu.copy()
        for c in cols:
            print('Normalizing column {}...'.format(c))
            norm_ruitu_df[c] = min_max_norm(selected_df_ruitu[c], ruitu_range_dic[c][0], ruitu_range_dic[c][1])

        print('OK! Has normalized for Ruitu dataframe!')
        
        # Fetch training and test data of numpy format   
        obs_input, obs_output = get_ndarray_by_sliding_window(norm_obs_df, input_len, output_len, 
                                                              vars_list=obs_var, only_target=obs_input_only_target_vars)
        
        _, ruitu_input = get_ndarray_by_sliding_window(norm_ruitu_df, input_len, output_len, 
                                                       vars_list=ruitu_var, only_target=False) # Always false
        
        #obs_input = np.expand_dims(obs_input, axis=1)
        #obs_output = np.expand_dims(obs_output, axis=1)
        #ruitu_input = np.expand_dims(ruitu_input, axis=1)

        print('obs_input shape:', obs_input.shape)
        print('obs_output (i.e., labels) shape:', obs_output.shape)
        print('ruitu_input shape:', ruitu_input.shape)       

        obs_inputs_.append(obs_input)
        ruitu_inputs_.append(ruitu_input)
        obs_outputs_.append(obs_output)

    obs_inputs_ = np.array(obs_inputs_)
    ruitu_inputs_ = np.array(ruitu_inputs_)
    obs_outputs_ = np.array(obs_outputs_)

    return {'obs_input':obs_inputs_, 
           'ruitu_input':ruitu_inputs_,
           'output_labels':obs_outputs_}

def load_pipeline(obs_df_file, ruitu_df_file, input_len=74, output_len=37, train_ratio=0.9, station_id=90001, only_target=True):
    
    print('The numbers of Obs varibles', len(obs_range_dic))
    print('The numbers of Ruitu varibles', len(ruitu_range_dic))

    # Define target variables
    targets=['t2m', 'rh2m', 'w10m']

    # Load filled Dataframe
    obs_df = load_pkl(processed_path, obs_df_file)
    ruitu_df = load_pkl(processed_path, ruitu_df_file)

    ruitu_df.reset_index(inplace=True)
    obs_df.reset_index(inplace=True)

    ruitu_df.set_index(['sta_id', 'time_index'], inplace=True)
    obs_df.set_index(['sta_id', 'time_index'], inplace=True)

    time_format_str='%Y-%m-%d %H:%M:%S'
    start_time = '2015-03-01 03:00:00'
    start_date = datetime.datetime.strptime(start_time, time_format_str)

    all_hours = 28512
    sta_id = station_id
    print('Selected Dataset of Station:', sta_id)
    selected_df_obs = obs_df.loc[sta_id]
    selected_df_ruitu = ruitu_df.loc[sta_id]

    selected_df_obs = reset_value_range(selected_df_obs, obs_range_dic)
    selected_df_ruitu = reset_value_range(selected_df_ruitu, ruitu_range_dic)

    # Max-min normalization
    # normalize for each column
    cols = selected_df_obs.columns
    norm_obs_df = selected_df_obs.copy()
    for c in cols:
        print('Normalizing column {}...'.format(c))
        norm_obs_df[c] = min_max_norm(selected_df_obs[c], obs_range_dic[c][0], obs_range_dic[c][1])

    print('OK! Has normalized for Observation dataframe!')

    # normalize for each column
    cols = selected_df_ruitu.columns
    norm_ruitu_df = selected_df_ruitu.copy()
    for c in cols:
        print('Normalizing column {}...'.format(c))
        norm_ruitu_df[c] = min_max_norm(selected_df_ruitu[c], ruitu_range_dic[c][0], ruitu_range_dic[c][1])

    print('OK! Has normalized for Ruitu dataframe!')

    # Fetch training and test data of numpy format
    
    train_obs_X, train_obs_Y , test_obs_X, test_obs_Y = get_train_test(norm_obs_df, input_len, output_len, per=train_ratio, data_name='obs', var_name = vars_names, only_target=only_target)
    
    train_ruitu_X, train_ruitu_Y , test_ruitu_X, test_ruitu_Y = get_train_test(norm_ruitu_df, input_len, output_len, per=train_ratio, data_name='ruitu', var_name = vars_names)
    
    print('Obs X shape:', train_obs_X.shape)
    print('Obs Y shape:', train_obs_Y.shape)
    print('Ruitu X shape:', train_ruitu_X.shape)
    print('Ruitu Y shape:',train_ruitu_Y.shape)
    
    return {'train_set':[train_obs_X, train_obs_Y, train_ruitu_X, train_ruitu_Y],
           'test_set':[test_obs_X, test_obs_Y, test_ruitu_X, test_ruitu_Y]}

if __name__ == '__main__':
    load_pipeline()