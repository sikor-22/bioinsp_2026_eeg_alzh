
from configparser import ConfigParser
import pandas as pd
import numpy as np


def parse_config(config_path):
    config = ConfigParser()
    config.read(config_path)
    ret = {}
    ret['filepath'] = config['data']['path']
    conditions = config['settings']['conditions'].split(",")
    conditions = [condition.strip() for condition in conditions]
    ret['conditions'] = conditions
    ret['binary'] = (int(config['settings']['binary_classification']) == 1)
    ret['drop_hc'] = (int(config['settings']['drop_hc_number']) == 1)
    print(ret['drop_hc'])
    return ret

def get_data(config):
    '''
    Parse filepath csv file into x, y.
    Args: filepath - path to dataset csv file.
    Rets: x, y - tuple of numpy arrays
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
