"""
# Synthetic IV Surface Data Generator
---

## Description

Generates a synthetic **implied volatility surface** over a grid of **strike × maturity** using the following parametric formula:

\\[
\\sigma(K, T) = a + b(K - K_{\\text{atm}})^2 + cT
\\]

Where:
- \\( a \\): base implied volatility level
- \\( b \\): controls the smile/skew curvature
- \\( c \\): controls the term structure slope
- \\( K_{\\text{atm}} \\): midpoint between `k_min` and `k_max`

---

## Function Signature

```python
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
)