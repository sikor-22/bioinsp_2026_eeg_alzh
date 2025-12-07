
from configparser import ConfigParser
import pandas as pd
import numpy as np


def parse_config(config_path):
    config = ConfigParser()
    config.read(config_path)
    ret = {}
    ret['root_path'] = config['data']['path']
    return ret

def get_data(filepath):
    '''
    Parse filepath csv file into x, y.
    Args: filepath - path to dataset csv file.
    Rets: x, y - tuple of numpy arrays
    '''
    data = pd.read_csv(filepath)

    subset = data[data['Condition'].isin(['HC3', 'AD'])]
    x, y = subset.loc[:, 'Fp1-1':'O2-AT_pdd'], subset['Condition']
    y = (y == 'AD').astype(int)
    y = np.array(y)
    x = np.array(x)
    return x, y
