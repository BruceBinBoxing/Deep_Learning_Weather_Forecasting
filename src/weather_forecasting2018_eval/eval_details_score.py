#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Weather Forecasting is a task of AI Challenger 全球AI挑战赛
This python script is used for calculating the accuracy of the test result,
based on your submited file and the reference file containing ground truth.
Usage:
python weather_forecasting2018_eval.py --submit SUBMIT_FILEPATH --obs OBSERVATION_FILEPATH --fore RMAPS_FILEPATH
A test case is provided, submited file is anene.csv, observation file is obs.csv, RMAPS result is anen.csv, test it by:
python weather_forecasting2018_eval.py --submit ./anen.csv --obs ./obs.csv --fore ./fore.csv
The accuracy of the submited result, error message and warning message will be printed.
"""


import time
import argparse
from math import sqrt
import numpy as np

import pandas as pd
from sklearn.metrics import mean_squared_error
from tabulate import tabulate

def bias(a, b):
    error = 0
    for i in a.index:
        error = error + (a[i] - b[i])
    err = error / len(a)
    return err


def rmse(a, b):
    return sqrt(mean_squared_error(a, b))


def score(a, b):
    return (b - a) / b


def score_bias(data_obs, data_fore):
    y_name = ['       t2m', '      rh2m', '      w10m']
    score = 0
    for i in y_name:
        score = score + bias(data_obs[i], data_fore[i])

    score = score / 3
    return score


def delete_non_value(data_obs, data_fore, data_anen, column):
    t = list(data_obs[data_obs[column] == -9999].index)
    data_obs_t2m = data_obs[column].drop(t)
    data_fore_t2m = data_fore[column].drop(t)
    data_anen_t2m = data_anen[column.strip()].drop(t)
    return data_obs_t2m, data_fore_t2m, data_anen_t2m


def _eval_result(fore_file, obs_file, anen_file):
    '''
    cal score
    :param fore_file: 超算结果
    :param obs_file: 正确答案
    :param anen_file: 选手提交结果
    :return:
    '''
    # eval the error rate

    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }

    try:
        data_obs = pd.read_csv(obs_file, encoding='gbk') # Obs label
        data_fore = pd.read_csv(fore_file, encoding='gbk') # Ruitu forecasting

        delimiter_list = [',', ';', ' ,', '    ', '  ', ' ', '\t']
        data_anen_columns_list = []
        for each_delimiter in delimiter_list:
            data_anen = pd.read_csv(anen_file, encoding='gbk', delimiter=each_delimiter) # Our prediction
            old_data_anen_columns_list = list(data_anen.columns)
            data_anen_columns_list = [_ for _ in old_data_anen_columns_list if _.strip() != '']
            if len(data_anen_columns_list) == 4:
                print('data_anen_columns_list:',data_anen_columns_list)
                break

        no_list = [_.strip() for _ in data_obs['  OBS_data']]
        #print(no_list)
        for each_no in no_list:
            if each_no.split('_')[1] in set(['00', '01', '02', '03']): # Omit the first 4 hours when calculating
                real_no = '  {each_no}'.format(**locals())
                this_index = list(data_obs[data_obs['  OBS_data'] == real_no].index)
                #print(this_index)
                data_obs = data_obs.drop(this_index)
                data_fore = data_fore.drop(this_index)
                data_anen = data_anen.drop(this_index)

        no_list_new = [_.strip() for _ in data_obs['  OBS_data']] # the station index after deleting the first 4 hours
        # 超算rmse
        data_anen_dict = {}
        for each_anen_column in data_anen_columns_list:
            data_anen_dict[each_anen_column.strip()] = data_anen[each_anen_column]

        stations=['90001', '90002','90003','90004','90005','90006','90007','90008','90009','90010', 'all']
        #stations=['90001', '90002','90003','90004','90005','90006','90007','90008','90009','90010']
        #stations=['all']
        #station_score = dict.fromkeys(stations)
        station_score = {'90001':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90002':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90003':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90004':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90005':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90006':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90007':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90008':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90009':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        '90010':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},
        'all':{'score':0, 't2m_score':0, 'rh2m_score':0, 'w10m_score':0},}

        data_anen_df = pd.DataFrame.from_dict(data_anen_dict)

        for s in stations:
            print("Evaluate station:", s)
            if s == 'all':
                One_S_obs = data_obs
                One_S_fore = data_fore
                One_S_anen = data_anen_df
            else:
                index_list=[]
                for each_no in no_list_new:
                    index_list.append(each_no.split('_')[0]==s)
                One_S_obs = data_obs.loc[index_list]
                One_S_fore = data_fore.loc[index_list]
                One_S_anen = data_anen_df.loc[index_list]

            data_obs_t2m, data_fore_t2m, data_anen_t2m = delete_non_value(One_S_obs, One_S_fore, One_S_anen, '       t2m')
            data_obs_rh2m, data_fore_rh2m, data_anen_rh2m = delete_non_value(One_S_obs, One_S_fore, One_S_anen, '      rh2m')
            data_obs_w10m, data_fore_w10m, data_anen_w10m = delete_non_value(One_S_obs, One_S_fore, One_S_anen, '      w10m')

            t2m_rmse = rmse(data_obs_t2m, data_fore_t2m)
            rh2m_rmse = rmse(data_obs_rh2m, data_fore_rh2m)
            w10m_rmse = rmse(data_obs_w10m, data_fore_w10m)

            # anenrmse
            t2m_rmse1 = rmse(data_obs_t2m, data_anen_t2m)
            rh2m_rmse1 = rmse(data_obs_rh2m, data_anen_rh2m)
            w10m_rmse1 = rmse(data_obs_w10m, data_anen_w10m)

            # 降低率得分
            score_all = (score(t2m_rmse1, t2m_rmse) + score(rh2m_rmse1, rh2m_rmse) + score(w10m_rmse1, w10m_rmse)) / 3
            # bias得分
            score_bias_fore = score_bias(data_obs, data_fore)

            result['score'] = score_all
            result['score_extra'] = {
                'bias_fore': score_bias_fore,
                't2m_rmse': t2m_rmse1,
                'rh2m_rmse': rh2m_rmse1,
                'w10m_rmse': w10m_rmse1
            }
            print('\n####### RMSE #######')
            print(tabulate([['MoE', t2m_rmse1, rh2m_rmse1, w10m_rmse1,],\
            ['Ruitu', t2m_rmse, rh2m_rmse, w10m_rmse]],\
            headers=['Method', 't2m', 'rh2m', 'w10m']))
            print('\n')
            print('####### Score #######')
            t2m_score = score(t2m_rmse1, t2m_rmse)
            rh2m_score = score(rh2m_rmse1, rh2m_rmse)
            w10m_score = score(w10m_rmse1, w10m_rmse)

            #print(tabulate([['MoE', t2m_score,\
            #rh2m_score, \
            #w10m_score]], \
            #headers=['Method', 't2m', 'rh2m', 'w10m']))

            print('\n')
            print('Station {} score: {}'.format(s, score_all))
            station_score[s]['score']=score_all
            station_score[s]['t2m_score']=t2m_score
            station_score[s]['rh2m_score']=rh2m_score
            station_score[s]['w10m_score']=w10m_score
            print('Bias:', score_bias_fore)

        print(station_score)
        print(tabulate([['90001', station_score['90001']['score'], station_score['90001']['t2m_score'], station_score['90001']['rh2m_score'], station_score['90001']['w10m_score']],\
        ['90002', station_score['90002']['score'], station_score['90002']['t2m_score'], station_score['90002']['rh2m_score'], station_score['90002']['w10m_score']],\
        ['90003', station_score['90003']['score'], station_score['90003']['t2m_score'], station_score['90003']['rh2m_score'], station_score['90003']['w10m_score']],\
        ['90004', station_score['90004']['score'], station_score['90004']['t2m_score'], station_score['90004']['rh2m_score'], station_score['90004']['w10m_score']],\
        ['90005', station_score['90005']['score'], station_score['90005']['t2m_score'], station_score['90005']['rh2m_score'], station_score['90005']['w10m_score']],\
        ['90006', station_score['90006']['score'], station_score['90006']['t2m_score'], station_score['90006']['rh2m_score'], station_score['90006']['w10m_score']],\
        ['90007', station_score['90007']['score'], station_score['90007']['t2m_score'], station_score['90007']['rh2m_score'], station_score['90007']['w10m_score']],\
        ['90008', station_score['90008']['score'], station_score['90008']['t2m_score'], station_score['90008']['rh2m_score'], station_score['90008']['w10m_score']],\
        ['90009', station_score['90009']['score'], station_score['90009']['t2m_score'], station_score['90009']['rh2m_score'], station_score['90009']['w10m_score']],\
        ['90010', station_score['90010']['score'], station_score['90010']['t2m_score'], station_score['90010']['rh2m_score'], station_score['90010']['w10m_score']],\
        ['all', station_score['all']['score'], station_score['all']['t2m_score'], station_score['all']['rh2m_score'], station_score['all']['w10m_score']]],\
        headers=['StationID', 'score', 't2m_score', 'rh2m_score', 'w10m_score']))

        #print("Average (Online) score:", np.mean(list(station_score.values())))
        #print("Average (Online) score:", station_score.values())
        #print(list(station_score.values()))

    except Exception as e:
        result['err_code'] = 1
        result['warning'] = str(e)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--submit',
        type=str,
        default='./anen.csv',
        help="""\
                Path to submited file\
            """
    )

    parser.add_argument(
        '--obs',
        type=str,
        default='./obs.csv',
        help="""
            Path to true result file
        """
    )

    parser.add_argument(
        '--fore',
        type=str,
        default='./fore.csv',
        help="""
            Path to RMAPS file
        """
    )

    args = parser.parse_args()
    start_time = time.time()
    result = _eval_result(args.fore, args.obs, args.submit)
    print('Running time:', time.time() - start_time)
    #print(result)
