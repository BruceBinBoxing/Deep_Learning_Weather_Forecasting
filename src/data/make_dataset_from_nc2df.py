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


class Transform():
    def __init__(self, input_filepath_name, output_filepath, process_phase):
        self.input_filepath_name = input_filepath_name
        self.output_filepath = output_filepath
        self.process_phase = process_phase
        self.NAN_ARRAY = np.ones(24)*(-9999) # If ruitu data are all NaNs
        self.obs_var_series_copy={} #Serve as a copy for using as replacement when ruitu_df exists NaN vectors.

    def read_nc_data(self):
        '''
        file_name: nc data file path
        '''
        nc_data = nc.Dataset(self.input_filepath_name)     # read nc-format file
        nc_dimensions, self.nc_variables= nc_data.dimensions, nc_data.variables   # 获取文件中的维度和变量
        #days_num = self.nc_variables.shape[0]

        self.var_obs = [] # Observation var name list
        self.var_ruitu=[] # Ruitu var name list

        for v in self.nc_variables: 
            if v.find("_obs") != -1:
                self.var_obs.append(v)
            if v.find('_M') != -1:
                self.var_ruitu.append(v)

        assert len(self.var_obs) == 9, 'Error, length of var_obs should be 9'
        assert len(self.var_ruitu) == 29, 'Error, length of var_ruitu should be 29'

        print('obs_var:',self.var_obs)
        print('obs_var numbers:', len(self.var_obs))

        print('ruitu_var:',self.var_ruitu)
        print('ruitu_var numbers:', len(self.var_ruitu))

        self.sta_id = self.nc_variables['station'][:].data
        self.foretime_index = self.nc_variables['foretimes'][:].data
        self.day_index = self.nc_variables['date'][:].data
        # Build a dictionary map key: station_id, value: from 0-9
        # Value of station_dic: {90001: 0, 90002: 1, 90003: 2, 90004: 3, 90005: 4, 90006: 5, 90007: 6, 90008: 7, 90009: 8, 90010: 9}
        self.station_dic ={}
        for i,s in enumerate(self.sta_id):
            self.station_dic[s]=i 
        assert self.station_dic == {90001: 0, 90002: 1, 90003: 2, 90004: 3, 90005: 4, 90006: 5, 90007: 6, 90008: 7, 90009: 8, 90010: 9}, 'Error of station_dic!'

        # Transform observation data
    def transform_and_save_data(self, var_type):
        df_list=[]
        assert var_type == 'obs' or var_type == 'ruitu', 'Error var_type can ONLY be obs OR ruitu!'
        
        if var_type == 'obs':
            var_list = self.var_obs
        elif var_type == 'ruitu':
            var_list = self.var_ruitu

        # Transform nc data to a dataframe for each station
        for skey in list(self.station_dic.keys())[:]:
        #for skey in [90001]:
            print("Generating a new station df for station: ",skey,'...')
            ### DataFrame Index
            df_one_sta=pd.DataFrame(columns=var_list)

            ### add time index series
            time_list=[]
            for d in list(self.day_index[:]):
                time_str = (str(d).split('.')[0])
                time_dt = datetime.datetime.strptime(time_str, '%Y%m%d%H')
                time_list.append(time_dt)
                for i in range(23):
                    left_sub_time = time_dt+pd.DateOffset(hours=(i+1))
                    time_list.append(left_sub_time)
                    
            #print(len(time_list))
            time_index_series = pd.Series(time_list)
            df_one_sta['time_index']=time_index_series
            df_one_sta['sta_id']=skey
            print('A new station df generated!')
            
            print('Fill the new df ...')
            ### add one var series for one station
            for i_v, v_name in enumerate(var_list):
                var_series=[]
                #self.obs_var_series_copy[v_name]=[] # initialize for the current variable

                click.echo('Adding the item: %s var series: %s in to %s_df_%s dataframe...'%(i_v, v_name, var_type, self.process_phase))
                var_data = self.nc_variables[v_name]
                for i in range(len(self.day_index)):
                    #if i<=1187:
                        #print(var_data[i,-13:, self.station_dic[skey]].data)
                        #print(var_data[i+1,:13, self.station_dic[skey]].data)
                        darray_24hours = var_data[i,:24, self.station_dic[skey]].data
                        var_series.extend(list(darray_24hours))
                        continue # No missing value imputation

                        if var_type == 'obs':
                            var_series.extend(list(darray_24hours)) # 3 is sta_id (all 10 stas)
                            #self.obs_var_series_copy[v_name].append(darray_24hours)

                        elif var_type == 'ruitu':
                            # If ruitu data are all NaNs, we fill that data by the counterpart observation data
                            if (darray_24hours == self.NAN_ARRAY).all():

                                if v_name in str(self.var_obs): # ONLY can replace vars of self.var_ruitu which are also the vars of self.var_obs. i.e., all include nine vars
                                    assert len(self.obs_var_series_copy[v_name]) == len(self.day_index), 'You must run trans_fn.transform_and_save_data("obs") first \
                                        then run trans_fn.transform_and_save_data("ruitu")'
                                    #darray_24hours = self.obs_var_series_copy[v_name][i]
                                    #print('Item index:',i, 'Fill 24-hours ruitu data by the counterpart observation data!')
                                    darray_24hours=self.NAN_ARRAY
                                    pass # No missing value imputation

                                else:
                                    # Set them to -9999
                                    darray_24hours=self.NAN_ARRAY
                                    # TODO: other imputation methods such as kNN
                                    pass

                            var_series.extend(list(darray_24hours))
                
                var_series = pd.Series(var_series)
                #var_series.name=v_name
                assert len(var_series) == len(self.day_index)*24, 'Error! length should be day_index*24=28512, but found {}'.format(len(var_series))

                df_one_sta[v_name]=var_series
            df_list.append(df_one_sta)

        aggregate_data_df = pd.concat(df_list)
        
        #save to .pkl file           
        if var_type == 'obs':
            #assert len(self.obs_var_series_copy)>0, 'Error!!!'
            save_pkl(aggregate_data_df, self.output_filepath, 'obs_df_{}.pkl'.format(self.process_phase))
        elif var_type == 'ruitu':
            save_pkl(aggregate_data_df, self.output_filepath, 'ruitu_df_{}.pkl'.format(self.process_phase))

def save_pkl(pkl_file, file_path, file_name, min_v=None, max_v=None):
    pk.dump(pkl_file, open(os.path.join(file_path, file_name), "wb"))
    print(file_name,' is dumped in: ', file_path)
    if((min_v is not None) and (max_v is not None)):
        pk.dump(min_v, open(os.path.join(file_path,file_name+"min"), "wb"))
        print(file_name+".min",' is dumped in: ', file_path)
        pk.dump(max_v, open(os.path.join(file_path,file_name+"max"), "wb"))
        print(file_name+".max",' is dumped in: ', file_path)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--process_phase', type=click.Choice(['train', 'val', 'testA', 'testB']))
@click.argument('output_filepath', type=click.Path(exists=True))
#@click.option('--sta_id', type=int, default=90001)

def main(input_filepath, process_phase, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    if process_phase == 'train':
        file_name='ai_challenger_wf2018_trainingset_20150301-20180531.nc'

    elif process_phase == 'val':
        file_name='ai_challenger_wf2018_validation_20180601-20180828_20180905.nc'

    elif process_phase == 'testA':
        file_name='ai_challenger_wf2018_testa1_20180829-20180924.nc'       
    elif process_phase == 'testB':
        file_name='ai_challenger_weather_testingsetB_20180829-20181015.nc'
    else:
        click.echo('Error! process_phase must be (train, validation or test)')

    logger.info('Transform raw data from %s to %s and %s' % (input_filepath+file_name, 
        output_filepath+'obs_df_'+process_phase+'.pkl',
        output_filepath+'ruitu_df_'+process_phase+'.pkl',))
    
    trans_fn = Transform(input_filepath+file_name, output_filepath, process_phase)
    trans_fn.read_nc_data()
    trans_fn.transform_and_save_data('obs')
    trans_fn.transform_and_save_data('ruitu')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()