
from configparser import ConfigParser
import pandas as pd
import numpy as np


def parse_config(config_path):
    '''
    Parse config ini file to a dict of program parameters
    :param config_path: Path to config .ini file
    '''
    config = ConfigParser()
    config.read(config_path)
    ret = {}

    ret['filepath'] = config['data']['path']
    conditions = config['data']['conditions'].split(",")
    conditions = [condition.strip() for condition in conditions]
    ret['conditions'] = conditions

    ret['binary'] = (int(config['settings']['binary_classification']) == 1)
    ret['drop_hc'] = (int(config['settings']['drop_hc_number']) == 1)

    ret['num_hidden_layers'] = int(config['model']['num_hidden_layers'])
    ret['hidden_size'] = int(config['model']['hidden_size'])
    ret['num_steps'] = int(config['model']['num_steps'])

    encoding_methods = config['model']['encoding_methods'].split(',')
    ret['encoding_methods'] = [enc_method.strip() for enc_method in encoding_methods]

    return ret

def get_data(config):
    '''
    Parse filepath csv file into x, y.
    :param config: config dictionary (most likely received from parse_config function)
    '''
    data = pd.read_csv(config['filepath'])
    condition_subset = config['conditions']
    if(config['binary']):
        condition_subset.remove('MCI')

    subset = data[data['Condition'].isin(config['conditions'])]
    x, A = subset.loc[:, 'Fp1-1':'O2-AT_pdd'], subset['Condition']

    if(config['drop_hc']):
        A = A.replace('HC3', 'HC')
        A = A.replace('HC2', 'HC')

    y = [sorted(set(A)).index(x) for x in A]
    y = np.array(y)
    y = y - np.min(y)
    x = np.array(x)
    return x, y
