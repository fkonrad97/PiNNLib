import torch
import numpy as np

def generate_iv_surface_data(
    num_strikes=10,
    num_maturities=10,
    k_min=80, k_max=120,
    t_min=0.1, t_max=2.0,
    base_vol=0.2,
    skew=0.0005,
    term_slope=0.05,
    device='cpu',
    flatten=True
):
    # Create strike and maturity grids
    strikes = torch.linspace(k_min, k_max, num_strikes)
    maturities = torch.linspace(t_min, t_max, num_maturities)

    K, T = torch.meshgrid(strikes, maturities, indexing='ij')
    K = K.to(device)
    T = T.to(device)

    K_atm = (k_min + k_max) / 2

    # Synthetic IV surface
    iv_surface = base_vol + skew * (K - K_atm)**2 + term_slope * T

    if flatten:
        features = torch.stack([K.flatten(), T.flatten()], dim=1)
        targets = iv_surface.flatten()
        return features, targets, (K, T, iv_surface)

    return K, T, iv_surface
