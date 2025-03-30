import torch
from torch.utils.data import Dataset
from pinnLib.transformer.data.synthetic.utils.black_scholes_data_generator_utils import generate_black_scholes_data

class BlackScholesDataset(Dataset):
    def __init__(self, n_samples=10000, device='cpu'):
        self.X, self.y = generate_black_scholes_data(n_samples=n_samples, device=device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
