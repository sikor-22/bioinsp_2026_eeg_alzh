import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import src.models as m
import src.utils as u
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
OUTPUT_DIR = 'output2'

def load_test_data():
    path = os.path.join(OUTPUT_DIR, 'test_data.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"CRITICAL: '{path}' not found. Run main.py first!")
    
    data = torch.load(path, map_location='cpu')
    return data['x_test'], data['y_test'], data['class_names'], data['input_size']

def plot_all(encoding):
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{encoding}.pth")
    if not os.path.exists(model_path):
        print(f"Skipping {encoding}, no model file found in {OUTPUT_DIR}.")
        return

    config = u.parse_config('config.ini') 
    x_test, y_test, class_names, input_size = load_test_data()
    num_classes = len(class_names)

    beta = 0.95 if encoding == 'latency' else 0.9
    threshold = 0.3 if encoding == 'latency' else 0.5


    model = m.SNNModelLearnable(input_size, 
                       int(config.get('num_hidden_layers', 1)),
                       int(config.get('hidden_size', 100)),
                       num_classes,
                       int(config.get('num_steps', 100)),
                       beta=beta, threshold=threshold)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print(f"Generating plots for {encoding}...")

    with torch.no_grad():
        spikes = m.spike_encoding(x_test, encoding, int(config.get('num_steps', 100)))
        spk_out, _ = model(spikes)
        output_rates = torch.sum(spk_out, dim=0)
        
        probs = torch.softmax(output_rates, dim=1).numpy()
        _, preds = torch.max(output_rates, 1)
        preds = preds.numpy()
        y_true = y_test.numpy()

    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({encoding.capitalize()})')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Paper_CM_{encoding}.png"), dpi=300)
    plt.close()

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({encoding.capitalize()})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Paper_ROC_{encoding}.png"), dpi=300)
    plt.close()

    hist_path = os.path.join(OUTPUT_DIR, f"history_{encoding}.pt")
    if os.path.exists(hist_path):
        hist = torch.load(hist_path)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(hist['train_loss'], color=color, lw=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Val Acc (%)', color=color)
        ax2.plot(hist['val_acc'], color=color, lw=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'Training Process ({encoding.capitalize()})')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Paper_History_{encoding}.png"), dpi=300)
        plt.close()

    print(f"Plots saved to {OUTPUT_DIR}/ for {encoding}.")

if __name__ == "__main__":
    plot_all('rate')
    plot_all('latency')