import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from sklearn.preprocessing import StandardScaler

class SNNModel(nn.Module):    
    def __init__(self, input_size, num_hidden_layers, hidden_layers_size, output_size, num_steps, beta=0.8, threshold = 0.2, learnable_beta = True, learnable_threshold = True):
        super(SNNModel, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.fclayers = []
        self.liflayers = []

        for i in range(num_hidden_layers):
            self.fclayers.append(nn.Linear(hidden_layers_size, hidden_layers_size))
            self.liflayers.append(snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold = threshold))

        self.fclayers = nn.ModuleList(self.fclayers)
        self.liflayers = nn.ModuleList(self.liflayers)
        
        self.fcfirst = nn.Linear(input_size, hidden_layers_size)
        self.liffirst = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold = threshold)
        
        self.fclast = nn.Linear(hidden_layers_size, output_size)
        self.liflast = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold = threshold)
        
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
            cur = self.fcfirst(x[step])
            spkfirst, memfirst = self.liffirst(cur, memfirst)
            spk_hidden = spkfirst

            for i in range(self.num_hidden_layers):
                cur = self.fclayers[i](spk_hidden)
                spk_hidden, mems[i] = self.liflayers[i](cur, mems[i])
            
            curlast = self.fclast(spk_hidden)
            spklast, memlast = self.liflast(curlast, memlast)

            spk_rec.append(spklast)
            mem_rec.append(memlast)

        
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    

class SNNSimpleModel(nn.Module):    
    def __init__(self, input_size, hidden_layers_size, output_size, num_steps, beta=0.8, threshold = 0.3):
        super(SNNSimpleModel, self).__init__()       
        self.fcfirst = nn.Linear(input_size, hidden_layers_size)
        self.liffirst = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold = threshold)

        self.fchidden = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.lifhidden = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold = threshold)
        
        self.fclast = nn.Linear(hidden_layers_size, output_size)
        self.liflast = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold = threshold)
        
        self.num_steps = num_steps
        self.hidden_size = hidden_layers_size

        nn.init.xavier_uniform_(self.fcfirst.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fchidden.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fclast.weight, gain=2.0)
        
    def forward(self, x):
        memfirst = self.liffirst.init_leaky()
        memhidden = self.lifhidden.init_leaky()
        memlast = self.liflast.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        for step in range(self.num_steps):
            cur = self.fcfirst(x[step])
            spkfirst, memfirst = self.liffirst(cur, memfirst)

            spkhidden = self.fchidden(spkfirst)
            spkhidden, memhidden = self.lifhidden(spkhidden, memhidden)

            curlast = self.fclast(spkhidden)
            spklast, memlast = self.liflast(curlast, memlast)

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

def spike_encoding(x, method, num_steps, gain = 0.45):
    '''
    Encode data to spike train
    
    :param x: Data to encode
    :param method: Encoding method, one of "rate", "latency"
    :param num_steps: Simulation steps
    :param gain: Gain for rate encoding
    '''
    # naive normalization, we will prescale before encoding, this is a failsafe
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / (x_max - x_min) # if x_max is equal to x_min we failed anyway
    if method == 'rate':
        # rate encoding: higher values = more frequent spikes
        spikes = spikegen.rate(
            x_normalized, 
            num_steps=num_steps, 
            gain=gain
        )
    elif method == 'latency':
        # latency encoding: higher values = earlier spikes
        spikes = spikegen.latency(
            x_normalized, 
            num_steps=num_steps, 
            threshold=0.05,
            normalize=True
        )
    else:
        raise ValueError(f"not a valid encoding method - {method}")
    
    return spikes


def preprocess_data(x, y):
    scaler = StandardScaler() # TODO: moze lepiej recznie to zrobic, nwm co to robi dokladnie
    x_scaled = scaler.fit_transform(x)
    y = torch.tensor(y).long()
    x = torch.tensor(x_scaled).float()
    return x, y


def debug_model_predictions(model, x_sample, y_sample, encoding_method, num_steps):
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        spikes = spike_encoding(x_sample.unsqueeze(0).to(device), encoding_method, num_steps)
        spk_rec, mem_rec = model(spikes)

        print(f"Total spikes in output layer: {torch.sum(spk_rec).item()}")
        print(f"Spike rate per neuron: {torch.mean(spk_rec.float()).item()}")
        
        # Check if any neurons are dead (no spikes)
        dead_neurons = torch.sum(spk_rec, dim=[0, 1]) == 0
        print(f"Dead neurons (no spikes): {torch.sum(dead_neurons).item()}/{dead_neurons.numel()}")
        
        # Check membrane potentials
        print(f"\nMembrane potential stats:")
        print(f"  Min: {torch.min(mem_rec).item():.4f}")
        print(f"  Max: {torch.max(mem_rec).item():.4f}")
        print(f"  Mean: {torch.mean(mem_rec).item():.4f}")
        print(f"  Std: {torch.std(mem_rec).item():.4f}")
        
        # Check output distribution
        output = torch.sum(mem_rec, dim=0)
        print(f"\nOutput logits (sum over time):")
        print(f"  Values: {output.squeeze().cpu().numpy()}")
        print(f"  Softmax: {torch.softmax(output, dim=1).squeeze().cpu().numpy()}")
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} - grad mean: {param.grad.mean().item():.6f}, std: {param.grad.std().item():.6f}")
            else:
                print(f"{name} - no gradient")


def train_model(x, y, x_val, y_val, model, num_epochs, num_steps, encoding_method, debug_prints = False, output_temp = 5.0):
    dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()


    if debug_prints:
        sample_idx = 0
        debug_model_predictions(model, x[sample_idx], y[sample_idx], 
                            encoding_method, num_steps)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            spikes = spike_encoding(batch_X, encoding_method, num_steps)
            spk_rec, mem_rec = model(spikes)
            
            # Train on membrane potentials
            output = torch.mean(mem_rec, dim=0)
            loss = loss_fn(output/output_temp, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                # training set
                x, y = x.to(device), y.to(device)
                spikes = spike_encoding(x, encoding_method, num_steps)
                spk_rec, mem_rec = model(spikes)
                output = torch.sum(spk_rec, dim=0)
                y_pred = torch.argmax(output, dim=1)
                accuracy = torch.sum(y_pred == y).float()/len(y)
                print(f"Epoch: {epoch} - Train Acc: {accuracy*100:.2f}%")
                # validation set
                x_val, y_val = x_val.to(device), y_val.to(device)
                spikes = spike_encoding(x_val, encoding_method, num_steps)
                spk_rec, mem_rec = model(spikes)
                output = torch.sum(spk_rec, dim=0)
                y_pred = torch.argmax(output, dim=1)
                val_accuracy = torch.sum(y_pred == y_val).float()/len(y_val)
                print(f"Epoch: {epoch} - Val Acc: {val_accuracy*100:.2f}%")
                print()

        if epoch % 10 == 0 and debug_prints:
            print(f"\n=== Debug at Epoch {epoch} ===")
            debug_model_predictions(model, batch_X[0], batch_y[0], 
                                   encoding_method, num_steps)