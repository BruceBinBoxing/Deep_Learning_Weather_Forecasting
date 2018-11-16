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
import sys

src_dir = os.path.join(os.getcwd(), 'src/data')
sys.path.append(src_dir)
from helper import save_pkl, load_pkl, min_max_norm
src_dir = os.path.join(os.getcwd(), 'src/models')
sys.path.append(src_dir)

# from competition_model_class import Seq2Seq_Class # during Debug and Developing
from seq2seq_class import Seq2Seq_Class # during the game and competition

def train(processed_path, train_data, val_data, model_save_path, model_name):
    train_dict = load_pkl(processed_path, train_data)
    val_dict = load_pkl(processed_path, val_data)


    print(train_dict.keys())
    print('Original input_obs data shape:')
    print(train_dict['input_obs'].shape)
    print(val_dict['input_obs'].shape)

    print('After clipping the 9 days, input_obs data shape:')
    train_dict['input_obs'] = train_dict['input_obs'][:,:-9,:,:]
    val_dict['input_obs'] = val_dict['input_obs'][:,:-9,:,:]
    print(train_dict['input_obs'].shape)
    print(val_dict['input_obs'].shape)

    enc_dec = Seq2Seq_Class(model_save_path=model_save_path,
                     model_structure_name=model_name, 
                     model_weights_name=model_name, 
                     model_name=model_name)
    enc_dec.build_graph()

    val_size=val_dict['input_ruitu'].shape[0] # 87 val samples
    val_ids=[]
    val_times=[]
    for i in range(10):
        val_ids.append(np.ones(shape=(val_size,37))*i)
    val_ids = np.stack(val_ids, axis=-1)
    print('val_ids.shape is:', val_ids.shape)
    val_times = np.array(range(37))
    val_times = np.tile(val_times,(val_size,1))
    print('val_times.shape is:',val_times.shape)

    enc_dec.fit(train_dict['input_obs'], train_dict['input_ruitu'], train_dict['ground_truth'],
           val_dict['input_obs'], val_dict['input_ruitu'], val_dict['ground_truth'], val_ids = val_ids, val_times=val_times,
            iterations=10000, batch_size=512, validation=True)

    print('Training finished!')

@click.command()
@click.argument('processed_path', type=click.Path(exists=True))
@click.option('--train_data', type=str)
@click.option('--val_data', type=str)
@click.argument('model_save_path', type=click.Path(exists=True))
@click.option('--model_name', type=str)

def main(processed_path, train_data, val_data, model_save_path, model_name):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    train(processed_path, train_data, val_data, model_save_path, model_name)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()