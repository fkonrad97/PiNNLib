import torch
import numpy as np
from scipy.stats import norm

def black_scholes_call_price(S, K, T, sigma, r):
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    call = S * torch.from_numpy(norm.cdf(d1.cpu().numpy())).float() - K * torch.exp(-r * T) * torch.from_numpy(norm.cdf(d2.cpu().numpy())).float()
    return call

def generate_black_scholes_data(n_samples=10000, device='cpu', r=0.01):
    torch.manual_seed(42)

    S = torch.FloatTensor(n_samples).uniform_(80, 120)   # spot
    K = torch.FloatTensor(n_samples).uniform_(80, 120)   # strike
    T = torch.FloatTensor(n_samples).uniform_(0.1, 2.0)  # time to maturity (in years)
    sigma = torch.FloatTensor(n_samples).uniform_(0.1, 0.5)  # volatility

    price = black_scholes_call_price(S, K, T, sigma, r)

    # Stack input features
    X = torch.stack([K, T, S, sigma], dim=1).to(device)
    y = price.to(device)

    return X, y
