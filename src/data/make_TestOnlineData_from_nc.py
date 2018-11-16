# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import netCDF4 as nc
import pickle as pk
import pandas as pd
import datetime
import os
import numpy as np
from helper import save_pkl, load_pkl, min_max_norm


obs_var=['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

target_var=['t2m_obs', 'rh2m_obs', 'w10m_obs']

ruitu_var=['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M',
       'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M',
       'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M',
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M',
       'Q850_M', 'Q700_M', 'Q500_M']

obs_range_dic={'psur_obs':[850,1100],
                't2m_obs':[-30,42], # Official value: [-20,42]
                'q2m_obs':[0,30],
                 'rh2m_obs':[0.0,100.0],
                 'w10m_obs':[0.0, 30.0],
                 'd10m_obs':[0.0, 360.0],
                 'u10m_obs':[-25.0, 20.0], # Official value: [-20,20]
                 'v10m_obs':[-20.0, 20.0],
                 'RAIN_obs':[0.0, 300.0],}

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

def netCDF2TheLastDay(data_file, phase_str, interim_filepath, datetime):
    '''
    phase_str: testA, testB or OnlineEveryDay
    '''
    data_dic={'input_obs':None,
         'input_ruitu':None,
         'ground_truth':None}

    print('processing...:', data_file)
    ori_data = nc.Dataset(data_file)     # 读取nc文件
    ori_dimensions, ori_variables= ori_data.dimensions, ori_data.variables   # 获取文件中的维度和变量
    date_index, fortime_index, station_index = 1, 2, 3    # 根据三个维度，读取该变量在特定日期、特定站点的特定预报时刻的数值
    var_obs = [] # var name list
    var_all =[]
    var_ruitu=[]
    for v in ori_variables:
        var_all.append(v)
        if v.find("_obs") != -1:
            var_obs.append(v)
        elif v.find('_M') != -1:
            var_ruitu.append(v)

    sta_id = ori_variables['station'][:].data
    print('sta_id:', sta_id)
    hour_index = ori_variables['foretimes'][:].data
    print('hour_index:', hour_index)
    day_index = ori_variables['date'][:].data
    print('day_index:', day_index)
    print(str(list(day_index)[-1]).split('.')[0])
    # build a map for staion and its index
    station_dic ={}
    for i,s in enumerate(sta_id):
        station_dic[s]=i
    print(station_dic)

    NUMS = ori_dimensions['date'].size
    print("The number of days:", NUMS)

    input_obs_dic = dict.fromkeys(var_obs, None)
    input_ruitu_dic = dict.fromkeys(var_ruitu, None)

    for v in var_obs:
        input_obs_dic[v] = ori_variables[v][-2,:,:].data[:-9] # Only select the last 28 days excluding the last 9 NaN days
        if (input_obs_dic[v] == -9999.).any():
            temp_df = pd.DataFrame(data=input_obs_dic[v])
            temp_df.replace(-9999., np.NaN, inplace=True)
            temp_df.interpolate(inplace=True)
            temp_df.bfill(inplace=True)
            temp_df.ffill(inplace=True)

            input_obs_dic[v] = temp_df.values

        assert not (input_obs_dic[v] == -9999.).any(), 'Error. -9999 happens in Obs for the predictive day!'
        assert not (input_obs_dic[v] == np.NaN).any(), 'Error. np.NaN happens in Obs for the predictive day!'

    for v in var_ruitu:
        input_ruitu_dic[v] = ori_variables[v][-1,:,:].data

        if (input_ruitu_dic[v] == -9999.).any():
            temp_df = pd.DataFrame(data=input_ruitu_dic[v])
            temp_df.replace(-9999., np.NaN, inplace=True)
            temp_df.interpolate(inplace=True)
            temp_df.bfill(inplace=True)
            temp_df.ffill(inplace=True)

            input_ruitu_dic[v] = temp_df.values

        assert not (input_ruitu_dic[v] == -9999.).any(), 'Error. -9999 happens in Ruitu for the predictive day!'
        assert not (input_ruitu_dic[v] == np.NaN).any(), 'Error. np.NaN happens in Obs for the predictive day!'

    data_dic['input_obs']=input_obs_dic
    data_dic['input_ruitu']=input_ruitu_dic

    save_pkl(data_dic, interim_filepath, '{}_one_predict_day_{}.dict'.format(phase_str, datetime))

    return '{}_one_predict_day_{}.dict'.format(phase_str, datetime)

def process_outlier_and_normalize(ndarray, max_min):
    '''
    Set outlier value into the normal range according to ruitu_range_dic AND obs_range_dic
    '''
    min_ = max_min[0]
    max_ = max_min[1]

    where_lower_min = ndarray < min_
    where_higher_max = ndarray > max_

    ndarray[where_lower_min]=min_
    ndarray[where_higher_max]=max_

    #normalize
    ndarray = min_max_norm(ndarray, min_, max_)

    return ndarray

def process_outlier_and_stack(interim_path, file_name, phase_str, datetime, processed_path):
    data_nc = load_pkl(interim_path, file_name)
    # Outlier processing
    for v in obs_var:
        data_nc['input_obs'][v] = process_outlier_and_normalize(data_nc['input_obs'][v], obs_range_dic[v])
    for v in ruitu_var:
        data_nc['input_ruitu'][v] = process_outlier_and_normalize(data_nc['input_ruitu'][v], ruitu_range_dic[v])

    stacked_data = [data_nc['input_obs'][v] for v in obs_var]
    stacked_input_obs = np.stack(stacked_data, axis=-1)

    stacked_data = [data_nc['input_ruitu'][v] for v in ruitu_var]
    stacked_input_ruitu = np.stack(stacked_data, axis=-1)

    print(stacked_input_obs.shape) #(sample_ind, timestep, station_id, features)
    print(stacked_input_ruitu.shape)

    data_dic={'input_obs':stacked_input_obs,
         'input_ruitu':stacked_input_ruitu}
    #normalize

    save_pkl(data_dic, processed_path, '{}_{}_norm.dict'.format(phase_str, datetime))



@click.command()
@click.argument('raw_filepath', type=click.Path(exists=True))
@click.option('--process_phase', type=click.Choice(['testA', 'testB', 'OnlineEveryDay']))
@click.argument('interim_filepath', type=click.Path(exists=True))
@click.argument('processed_path', type=click.Path(exists=True))
@click.option('--datetime', type=int)

#@click.option('--sta_id', type=int, default=90001)

def main(raw_filepath, process_phase, interim_filepath, datetime, processed_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    if process_phase == 'testA':
        file_name='ai_challenger_wf2018_testa1_20180829-20180924.nc'

    elif process_phase == 'testB':
        file_name='ai_challenger_weather_testingsetB_20180829-20181015.nc'

    elif process_phase == 'OnlineEveryDay':
        file_name='ai_challenger_wf2018_testb1_20180829-20181028.nc'
        #click.echo('Error! process_phase must be (testA, testB or OnlineEveryDay)')

    interim_file_name = netCDF2TheLastDay(raw_filepath+file_name, process_phase, interim_filepath, datetime)
    process_outlier_and_stack(interim_filepath, interim_file_name, process_phase, datetime, processed_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
