import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from src.data_loader import load_movielens_data
from src.models.ncf_mlp import NCF_MLP
from src.models.autoencoder import AutoencoderCF
from src.train import train_ncf, train_autoencoder

def evaluate_model(model, test_loader, model_type, num_users, num_items, device='cpu'):
    model.eval()
    total_mae = 0
    count = 0
    with torch.no_grad():
        if model_type == 'ncf':
            for user_ids, item_ids, ratings in test_loader:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                predictions = model(user_ids, item_ids)
                mae = torch.abs(predictions - ratings).mean().item()
                total_mae += mae * len(ratings)
                count += len(ratings)
        elif model_type == 'autoencoder':
            user_item_matrix = torch.zeros(num_users, num_items).to(device)
            for user_ids, item_ids, ratings in test_loader.dataset:
                user_item_matrix[user_ids, item_ids] = ratings
            reconstructed = model(user_item_matrix)
            for user_ids, item_ids, ratings in test_loader:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                predictions = reconstructed[user_ids, item_ids]
                mae = torch.abs(predictions - ratings).mean().item()
                total_mae += mae * len(ratings)
                count += len(ratings)
    return total_mae / count

def save_results(results, filename='results/results.txt'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('Model\tMAE\n')
        for model_name, mae in results.items():
            f.write(f'{model_name}\t{mae:.4f}\n')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, num_users, num_items = load_movielens_data()
    
    ncf_model = NCF_MLP(num_users, num_items).to(device)
    ncf_model = train_ncf(ncf_model, train_loader, device=device)
    ncf_mae = evaluate_model(ncf_model, test_loader, 'ncf', num_users, num_items, device=device)
    
    autoencoder_model = AutoencoderCF(num_items).to(device)
    autoencoder_model = train_autoencoder(autoencoder_model, train_loader, num_users, num_items, device=device)
    autoencoder_mae = evaluate_model(autoencoder_model, test_loader, 'autoencoder', num_users, num_items, device=device)
    
    results = {'NCF_MLP': ncf_mae, 'Autoencoder': autoencoder_mae}
    save_results(results)