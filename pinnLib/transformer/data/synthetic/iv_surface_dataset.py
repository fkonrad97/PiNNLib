import torch
from torch.utils.data import Dataset
from pinnLib.transformer.data.synthetic.utils.generate_iv_surface_data import generate_iv_surface_data

class IVSurfaceDataset(Dataset):
    def __init__(
        self,
        num_strikes=10,
        num_maturities=10,
        k_min=80,
        k_max=120,
        t_min=0.1,
        t_max=2.0,
        base_vol=0.2,
        skew=0.0005,
        term_slope=0.05,
        device='cpu'
    ):
        self.features, self.targets, _ = generate_iv_surface_data(
            num_strikes=num_strikes,
            num_maturities=num_maturities,
            k_min=k_min,
            k_max=k_max,
            t_min=t_min,
            t_max=t_max,
            base_vol=base_vol,
            skew=skew,
            term_slope=term_slope,
            device=device,
            flatten=True
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
