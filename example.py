'''
WIP for how to use
'''

import src.utils as u
import src.models as m
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

config = u.parse_config('config.ini')
x, y = u.get_data(config)

print(f"HC is {np.sum(y==1)/len(y)} of dataset :(")

model = m.get_model(x.shape[1], len(set(y)), config)
x, y = m.preprocess_data(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


m.train_model(x_train, y_train, model, num_epochs=50, encoding_method='rate', num_steps=config['num_steps'])