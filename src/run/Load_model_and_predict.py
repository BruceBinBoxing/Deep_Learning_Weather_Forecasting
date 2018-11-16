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

from keras import backend as K

src_dir = os.path.join(os.getcwd(), 'src/data')
sys.path.append(src_dir)
src_dir = os.path.join(os.getcwd(), 'src/models')
sys.path.append(src_dir)

from helper import save_pkl, load_pkl, min_max_norm, renorm_for_submit, renorm, predict
from competition_model_class import Seq2Seq_Class
from keras.models import load_model, model_from_json
from keras.utils.generic_utils import get_custom_objects
def swish(x):
        return (K.sigmoid(x) * x)
get_custom_objects().update({'swish':swish})

def Load_and_predict(model_save_path, model_name, processed_path, test_file_name, saved_csv_path, saved_csv_name):
    
    #TODO: delete class!
    
    # load json and create model
    json_file = open(model_save_path+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    print(model.summary())
    # load weights into new model
    model.load_weights(model_save_path+model_name+'.h5')

    ## Load test data
    test_file= test_file_name
    test_data= load_pkl(processed_path, test_file)

    test_inputs = test_data['input_obs']
    test_ruitu = test_data['input_ruitu']

    test_inputs = np.expand_dims(test_inputs, axis=0)
    test_ruitu = np.expand_dims(test_ruitu, axis=0)
    #add test ids
    test_ids=[]
    for i in range(10):
        test_ids.append(np.ones(shape=(1,37))*i)
    test_ids = np.stack(test_ids, axis=-1)
    # add time
    test_size = test_inputs.shape[0]
    test_times = np.array(range(37))
    test_times = np.tile(test_times,(test_size,1))

    pred_result, pred_var_result = predict(model, test_inputs, test_ruitu, test_ids, test_times)

    print(pred_result.shape)
    print(pred_var_result.shape)

    ### save the result for submit
    df_empty = renorm_for_submit(pred_mean=pred_result[0], pred_var=pred_var_result[0], ruitu_inputs=test_ruitu[0],
                                        timestep_to_ensemble=21, alpha=1)

    df_empty = df_empty.rename(columns={"t2m":"       t2m", 
                             "rh2m":"      rh2m",
                            "w10m":"      w10m"})

    save_path = saved_csv_path
    df_empty.to_csv(path_or_buf=save_path+saved_csv_name, header=True, index=False)
    print('Ok! You can submit now!')

@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--model_name', type=str)
@click.argument('processed_path', type=click.Path(exists=True))
@click.option('--test_file_name', type=str)
@click.option('--saved_csv_path', type=str)
@click.option('--saved_csv_name')
def main(model_path, model_name, processed_path, test_file_name, saved_csv_path, saved_csv_name):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    Load_and_predict(model_path, model_name, processed_path, test_file_name, saved_csv_path, saved_csv_name)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()