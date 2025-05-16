import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os

class MovieLensDataset(Dataset):
    def __init__(self, ratings, use_idx=False):
        if use_idx:
            self.users = torch.tensor(ratings['user_idx'].values, dtype=torch.long)
            self.items = torch.tensor(ratings['movie_idx'].values, dtype=torch.long)
        else:
            self.users = torch.tensor(ratings['userId'].values, dtype=torch.long)
            self.items = torch.tensor(ratings['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def load_movielens_data(data_path=None):
    if data_path is None:
        data_path = os.path.join('data', 'movielens_1m', 'ratings.dat')
    columns = ['userId', 'movieId', 'rating', 'timestamp']
    data = pd.read_csv(data_path, sep='::', names=columns, engine='python')
    # Map userId and movieId to contiguous indices
    user_id_map = {id_: idx for idx, id_ in enumerate(data['userId'].unique())}
    movie_id_map = {id_: idx for idx, id_ in enumerate(data['movieId'].unique())}
    data['user_idx'] = data['userId'].map(user_id_map)
    data['movie_idx'] = data['movieId'].map(movie_id_map)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = MovieLensDataset(train_data, use_idx=True)
    test_dataset = MovieLensDataset(test_data, use_idx=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader, len(user_id_map), len(movie_id_map)