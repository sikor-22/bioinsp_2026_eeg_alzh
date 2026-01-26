import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SNNModel(nn.Module):
    """
    Spiking Neural Network with learnable LIF neuron parameters.
    
    Args:
        input_size: Number of input features
        num_hidden_layers: Number of hidden layers
        hidden_layers_size: Size of each hidden layer
        output_size: Number of output classes
        num_steps: Number of time steps for simulation
        beta: decay rate
        threshold: spike threshold
    """   
    def __init__(self, input_size, num_hidden_layers, hidden_layers_size, output_size, num_steps, beta=0.9, threshold = 0.5):
        super(SNNModel, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.fclayers = []
        self.liflayers = []
        
        self.fcfirst = nn.Linear(input_size, hidden_layers_size)
        self.liffirst = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold=threshold)
        
        for i in range(num_hidden_layers):
            self.fclayers.append(nn.Linear(hidden_layers_size, hidden_layers_size))
            self.liflayers.append(snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold=threshold))

        self.fclayers = nn.ModuleList(self.fclayers)
        self.liflayers = nn.ModuleList(self.liflayers)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fclast = nn.Linear(hidden_layers_size, output_size)
        self.liflast = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), threshold=threshold, output=True)
        
        self.num_steps = num_steps
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [num_steps, batch_size, input_size]
            
        Returns:
            spk_rec: Spike recordings of shape [num_steps, batch_size, output_size]
            mem_rec: Membrane potential recordings of shape [num_steps, batch_size, output_size]
        """
        
        memfirst = self.liffirst.init_leaky()
        mems = [lif.init_leaky() for lif in self.liflayers]
        memlast = self.liflast.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        for step in range(self.num_steps):
            inp = x[step]
            cur = self.fcfirst(inp)
            spkfirst, memfirst = self.liffirst(cur, memfirst)
            spk_hidden = self.dropout(spkfirst)

            for i in range(self.num_hidden_layers):
                cur = self.fclayers[i](spk_hidden)
                spk_hidden, mems[i] = self.liflayers[i](cur, mems[i])
                spk_hidden = self.dropout(spk_hidden)
            
            curlast = self.fclast(spk_hidden)
            spklast, memlast = self.liflast(curlast, memlast)

            spk_rec.append(spklast)
            mem_rec.append(memlast)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class SNNModelLearnable(nn.Module):
    """
    Spiking Neural Network with learnable LIF neuron parameters.
    
    Args:
        input_size: Number of input features
        num_hidden_layers: Number of hidden layers
        hidden_layers_size: Size of each hidden layer
        output_size: Number of output classes
        num_steps: Number of time steps for simulation
        beta: Initial decay rate (learnable)
        threshold: Initial spike threshold (learnable)
    """
    
    def __init__(self, input_size, num_hidden_layers, hidden_layers_size, output_size, num_steps, beta=0.9, threshold=0.5):
        super(SNNModelLearnable, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_steps = num_steps
        
        beta_init = torch.tensor([beta], dtype=torch.float32)
        threshold_init = torch.tensor([threshold], dtype=torch.float32)
        
        self.fcfirst = nn.Linear(input_size, hidden_layers_size)
        self.liffirst = snn.Leaky(beta=nn.Parameter(beta_init.clone()), 
                                   spike_grad=surrogate.fast_sigmoid(), 
                                   threshold=nn.Parameter(threshold_init.clone()))
        
        self.fclayers = nn.ModuleList([
            nn.Linear(hidden_layers_size, hidden_layers_size) 
            for _ in range(num_hidden_layers)
        ])
        self.liflayers = nn.ModuleList([
            snn.Leaky(beta=nn.Parameter(beta_init.clone()), 
                      spike_grad=surrogate.fast_sigmoid(), 
                      threshold=nn.Parameter(threshold_init.clone()))
            for _ in range(num_hidden_layers)
        ])
        
        self.dropout = nn.Dropout(0.5)
        
        self.fclast = nn.Linear(hidden_layers_size, output_size)
        self.liflast = snn.Leaky(beta=nn.Parameter(beta_init.clone()), 
                                  spike_grad=surrogate.fast_sigmoid(), 
                                  threshold=nn.Parameter(threshold_init.clone()),
                                  output=True)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [num_steps, batch_size, input_size]
            
        Returns:
            spk_rec: Spike recordings of shape [num_steps, batch_size, output_size]
            mem_rec: Membrane potential recordings of shape [num_steps, batch_size, output_size]
        """
        memfirst = self.liffirst.init_leaky()
        mems = [lif.init_leaky() for lif in self.liflayers]
        memlast = self.liflast.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        for step in range(self.num_steps):
            inp = x[step]
            cur = self.fcfirst(inp)
            spkfirst, memfirst = self.liffirst(cur, memfirst)
            spk_hidden = self.dropout(spkfirst)

            for i in range(self.num_hidden_layers):
                cur = self.fclayers[i](spk_hidden)
                spk_hidden, mems[i] = self.liflayers[i](cur, mems[i])
                spk_hidden = self.dropout(spk_hidden)
            
            curlast = self.fclast(spk_hidden)
            spklast, memlast = self.liflast(curlast, memlast)

            spk_rec.append(spklast)
            mem_rec.append(memlast)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)


def get_model(input_size, output_size, config):
    '''
    Get SNNModel object
    '''
    model = SNNModel(input_size = input_size,
                    num_hidden_layers=config.get('num_hidden_layers', 1),
                    hidden_layers_size=config.get('hidden_size', 100),
                    output_size=output_size,
                    num_steps=config.get('num_steps', 25))
    return model

def spike_encoding(x, method, num_steps, gain = 0.5):
    '''
    Encode data to spike train. 
    ASSUMES X IS ALREADY NORMALIZED appropriately (e.g. 0-1 for latency).
    '''
    
    if method == 'rate':
        spikes = spikegen.rate(x, num_steps=num_steps, gain=gain)
    elif method == 'latency':
        spikes = spikegen.latency(x, num_steps=num_steps, tau=5, threshold=0.01, normalize=True, clip=True)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return spikes

def preprocess_data(x, y):
    scaler = MinMaxScaler() 
    x_scaled = scaler.fit_transform(x)
    
    y = torch.tensor(y).long()
    x = torch.tensor(x_scaled).float()
    return x, y

def train_model(x_train, y_train, x_val, y_val, model, num_epochs, num_steps, encoding_method, save_path='best_model.pth', class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)

    wd = 1e-2 if encoding_method == 'rate' else 1e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=wd)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            spikes_in = spike_encoding(batch_x, encoding_method, num_steps)
            spk_out, mem_out = model(spikes_in)
            output_logits = torch.mean(mem_out, dim=0)
            
            loss = loss_fn(output_logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                spikes_in = spike_encoding(batch_x, encoding_method, num_steps)
                spk_out, mem_out = model(spikes_in)
            
                output_rates = torch.sum(spk_out, dim=0)
                _, predicted = torch.max(output_rates, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * correct / total
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} \t Loss: {avg_loss:.4f} \t Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            # print("Saved best model")
            
    print(f"Training finished. Best Val Acc: {best_val_acc:.2f}%")
    return history