import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.ncf_mlp import NCF_MLP
from src.models.autoencoder import AutoencoderCF

def train_ncf(model, train_loader, num_epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user_ids, item_ids, ratings in train_loader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'NCF Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')
    return model

def train_autoencoder(model, train_loader, num_users, num_items, num_epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    user_item_matrix = torch.zeros(num_users, num_items).to(device)
    for user_ids, item_ids, ratings in train_loader.dataset:
        user_item_matrix[user_ids, item_ids] = ratings
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed = model(user_item_matrix)
        loss = criterion(reconstructed, user_item_matrix)
        loss.backward()
        optimizer.step()
        print(f'Autoencoder Epoch {epoch+1}, Loss: {loss.item():.4f}')
    return model