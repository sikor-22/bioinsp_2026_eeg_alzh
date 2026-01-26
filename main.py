import src.utils as u
import src.models as m
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import torch
import numpy as np
import os

def main():
    OUTPUT_DIR = 'output2'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Outputs will be saved to: {OUTPUT_DIR}/")

    config_path = 'config.ini'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path}")
    
    config = u.parse_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configuration loaded. Running on: {device}")
    
    print("Loading raw data...")
    x, y = u.get_data(config)
    
    # Podział
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Initial Split -> Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val   = scaler.transform(x_val)
    x_test  = scaler.transform(x_test)

    k = int(config.get('feature_selection', {}).get('num_features', 100))
    print(f"Selecting top {k} features on ORIGINAL data...")
    
    selector = SelectKBest(score_func=f_classif, k=k)
    x_train = selector.fit_transform(x_train, y_train)
    x_val   = selector.transform(x_val)
    x_test  = selector.transform(x_test)

    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train) # Nadpisujemy zmienne
    print(f"Class distribution after SMOTE:  {np.bincount(y_train)}")
   
    x_train = np.clip(x_train, 0, 1)
    x_val   = np.clip(x_val, 0, 1)
    x_test  = np.clip(x_test, 0, 1)


    test_data_path = os.path.join(OUTPUT_DIR, 'test_data.pt')
    torch.save({
        'x_test': torch.tensor(x_test).float(),
        'y_test': torch.tensor(y_test).long(),
        'class_names': ['AD', 'HC', 'MCI'],
        'input_size': k
    }, test_data_path)
    print(f"Test data saved to '{test_data_path}'.")


    train_dataset = (torch.tensor(x_train).float().to(device), torch.tensor(y_train).long().to(device))
    val_dataset   = (torch.tensor(x_val).float().to(device),   torch.tensor(y_val).long().to(device))
    

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights).float().to(device)


    num_classes = len(classes)
    
    for encoding in config['encoding_methods']:
        print(f"\n--- Training {encoding.upper()} Encoding ---")
        
        beta = 0.95 if encoding == 'latency' else 0.9
        threshold = 0.3 if encoding == 'latency' else 0.5
        
        model = m.SNNModelLearnable(
            input_size=k, 
            num_hidden_layers=int(config['num_hidden_layers']),
            hidden_layers_size=int(config['hidden_size']),
            output_size=num_classes,
            num_steps=int(config['num_steps']),
            beta=beta,
            threshold=threshold
        ).to(device)
        
        model_save_path = os.path.join(OUTPUT_DIR, f"best_model_{encoding}.pth")
        
        history = m.train_model(
            train_dataset[0], train_dataset[1],
            val_dataset[0],   val_dataset[1],
            model, 
            num_epochs=50, 
            num_steps=int(config['num_steps']), 
            encoding_method=encoding,
            save_path=model_save_path,
            class_weights=class_weights
        )

        torch.save(history, os.path.join(OUTPUT_DIR, f"history_{encoding}.pt"))

if __name__ == "__main__":
    main()