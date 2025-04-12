import torch
import matplotlib.pyplot as plt
from pinnLib.transformer.model.option_models.iv_surface_transformer_model import IVSurfaceTransformerModel
from pinnLib.transformer.data.synthetic.utils.generate_iv_surface_data import generate_iv_surface_data

# --- Model config (must match training) ---
embed_dim = 64
num_heads = 4
ff_hidden_dim = 128
num_layers = 4
head_hidden_dim = 64
num_strikes = 10
num_maturities = 10
input_dim = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Generate new surface for testing ---
X, y_true, (K, T, surface_true) = generate_iv_surface_data(
    num_strikes=num_strikes,
    num_maturities=num_maturities,
    base_vol=0.2,
    skew=0.001,
    term_slope=0.05,
    flatten=True,
    device=device
)

X = X.unsqueeze(0)  # (1, seq_len, input_dim)

# --- Load trained model ---
model = IVSurfaceTransformerModel(
    input_dim=input_dim,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    num_strikes=num_strikes,
    num_maturities=num_maturities,
    head_hidden_dim=head_hidden_dim
).to(device)

model.load_state_dict(torch.load("iv_surface_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    pred_surface = model(X).squeeze(0).cpu()  # (num_strikes, num_maturities)

# --- Plot predicted vs true ---
fig = plt.figure(figsize=(12, 5))

# True IV surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(K.cpu(), T.cpu(), surface_true.cpu(), cmap='viridis')
ax1.set_title('True IV Surface')
ax1.set_xlabel('Strike')
ax1.set_ylabel('Maturity')

# Predicted IV surface
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(K.cpu(), T.cpu(), pred_surface.cpu(), cmap='plasma')
ax2.set_title('Predicted IV Surface')
ax2.set_xlabel('Strike')
ax2.set_ylabel('Maturity')

plt.tight_layout()
plt.show()