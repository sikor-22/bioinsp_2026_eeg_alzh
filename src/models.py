import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SNNModel(nn.Module):    
    def __init__(self, input_size, num_hidden_layers, hidden_layers_size, output_size, num_steps, beta=0.5):
        super(SNNModel, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.fclayers = []
        self.liflayers = []

        for i in range(num_hidden_layers):
            self.fclayers.append(nn.Linear(hidden_layers_size, hidden_layers_size))
            self.liflayers.append(snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid()))
        
        self.fcfirst = nn.Linear(input_size, hidden_layers_size)
        self.liffirst = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.fclast = nn.Linear(hidden_layers_size, output_size)
        self.liflast = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.num_steps = num_steps
        self.hidden_size = hidden_layers_size
        
    def forward(self, x):
        memfirst = self.liffirst.init_leaky()
        mems = []
        for lif in self.liflayers:
            mems.append(lif.init_leaky())
        memlast = self.liflast.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        for step in range(self.num_steps):
            cur = self.fcfirst(x)
            spkfirst, memfirst = self.liffirst(cur, memfirst)
            spk_hidden = spkfirst

            for i in range(self.num_hidden_layers):
                cur = self.fclayers[i](spk_hidden)
                spk_hidden, mems[i] = self.liflayers(cur, mems[i])
            
            curlast = self.fc3(spk_hidden)
            spklast, memlast = self.lif3(curlast, memlast)
            
            spk_rec.append(spklast)
            mem_rec.append(memlast)
            
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    

def get_model(input_size, output_size, config):
    '''
    Get SNNModel object
    
    :param input_size: size of input to the model
    :param output_size: size of model output
    :param config: config dict
    '''
    model = SNNModel(input_size = input_size,
                    num_hidden_layers=config['num_hidden_layers'],
                    hidden_layers_size=config['hidden_size'],
                    output_size=output_size,
                    num_steps=config['num_steps'])
    return model