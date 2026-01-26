
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

    if 'feature_selection' in config:
        ret['feature_selection'] = {}
        if 'num_features' in config['feature_selection']:
            ret['feature_selection']['num_features'] = int(config['feature_selection']['num_features'])
        if 'method' in config['feature_selection']:
            ret['feature_selection']['method'] = config['feature_selection']['method']

    return ret

def get_data(config):
    '''
    Parse filepath csv file into x, y.
    :param config: config dictionary (most likely received from parse_config function)
    '''
    filepath = config['filepath']
            
    data = pd.read_csv(filepath)
    condition_subset = config['conditions'][:]
    if config['binary']:
        if 'MCI' in condition_subset:
            condition_subset.remove('MCI')
            
    subset = data[data['Condition'].isin(condition_subset)]
    try:
        if 'Fp1-1' in subset.columns:
             x_cols = [c for c in subset.columns if c != 'Condition' and c != 'Subject' and c != 'Group' and c != 'Gender' and c != 'Age' and c != 'MMSE']
             x = subset.select_dtypes(include=[np.number])
        else:
             x = subset.select_dtypes(include=[np.number])
    except Exception as e:
        print(f"Error selecting columns: {e}")
        x = subset.select_dtypes(include=[np.number])

    A = subset['Condition']
    if(config['drop_hc']):
        A = A.replace('HC3', 'HC').replace('HC2', 'HC').replace('HC1', 'HC')

    x = x.astype(np.float64)
    x = x.dropna(how='all', axis=1)
    x = x.fillna(0)
    
    # mapping: AD=0, HC=1, MCI=2
    unique_labels = sorted(set(A))
    print(f"Classes found: {unique_labels}")
    
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in A])
    
    x = np.array(x)
    return x, y

