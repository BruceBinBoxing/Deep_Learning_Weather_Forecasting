# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pickle
import pandas as pd
import sys
import datetime

from helper import *
from data_load import transform_from_df2ndarray, reset_value_range

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--obs_file_name')
@click.option('--ruitu_file_name')
@click.option('--station_id', type=int) # 999 means all stations 
@click.argument('output_filepath', type=click.Path(exists=True))

def main(input_filepath, obs_file_name, ruitu_file_name, station_id, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Missing value imputation from %s%s'%(input_filepath,obs_file_name))
    assert obs_file_name.split('.')[0].split('_')[-1] == ruitu_file_name.split('.')[0].split('_')[-1], "Error! Both file names must have the same phase info (train, val, test)"
    # Obs data imputation
    obs_df = load_pkl(input_filepath, obs_file_name)
    obs_df.set_index(['sta_id', 'time_index'], inplace=True)
    obs_df.replace(-9999., np.NaN, inplace=True)
    print('Simply replace -9999. with np.NaN!')
    obs_df.fillna(method='ffill', inplace=True)
    obs_df.fillna(method='bfill', inplace=True)

    # Ruitu data imputation
    logger.info('Missing value imputation from %s%s'%(input_filepath,ruitu_file_name))
    df_ruitu = load_pkl(input_filepath, ruitu_file_name)
    df_ruitu.set_index(['sta_id', 'time_index'], inplace=True)
    df_ruitu.replace(-9999., np.NaN, inplace=True)
    print('Simply replace -9999. with np.NaN!')
    df_ruitu.fillna(method='bfill', inplace=True)
    df_ruitu.fillna(method='ffill', inplace=True)

    # Make ndarray data for deep models
    logger.info('Transform imputed dataframes to ndarray ...')    
    ndarray_dic= transform_from_df2ndarray(obs_df=obs_df, ruitu_df=df_ruitu, 
        input_len=74, output_len=37, station_id=station_id, obs_input_only_target_vars=False)

    phase_flag = obs_file_name.split('.')[0].split('_')[-1] # train, val, OR test
    assert phase_flag == 'train' or phase_flag == 'test' or phase_flag == 'val', 'phase can only be train, val, OR test; Please reproduce from scrach by running MakeFile script...!'

    save_pkl(ndarray_dic, output_filepath, '{}_sta_{}.ndarray'.format(phase_flag, station_id))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()