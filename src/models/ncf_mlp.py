import torch
import torch.nn as nn

class NCF_MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NCF_MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            input_dim = hidden_dim
        layers += [nn.Linear(input_dim, 1), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        interaction = torch.cat([user_emb, item_emb], dim=-1)
        output = self.mlp(interaction).squeeze() * 5.0
        return output