import torch

def make_rbf_centers_batch(k, a, b):
    """
    a, b: (in_dim,) tensors
    returns: (in_dim, k) centers
      k=1 -> center at midpoint (a+b)/2
      k=2 -> centers at [a, b]
      k>2 -> centers at [a, ..., b] (k-2 uniform interior)
    """
    if k == 1:
        return ((a + b) / 2.0).unsqueeze(-1)
    t = torch.linspace(0.0, 1.0, k, device=a.device, dtype=a.dtype)
    return a[:, None] + (b - a)[:, None] * t[None, :]

def chebyshev_basis(x, x_min, x_max, degree):
    x_range = (x_max - x_min).clamp(min=1e-8)
    x_norm = 2.0 * (x - x_min) / x_range - 1.0
    x_norm = torch.clamp(x_norm, -1.0, 1.0)
    T = [torch.ones_like(x_norm)]
    if degree >= 1:
        T.append(x_norm)
    for n in range(2, degree + 1):
        T.append(2.0 * x_norm * T[-1] - T[-2])
    return torch.stack(T, dim=-1)

def rbf_basis(x, centers, sigma):
    """
    x: (batch, in_dim)
    centers: (in_dim, n_centers)
    sigma: 0-dim tensor | (in_dim,) | (in_dim, n_centers)
    """
    diff = x[:, :, None] - centers[None, :, :]
    if isinstance(sigma, torch.Tensor):
        if sigma.dim() == 1:
            sigma = sigma[None, :, None]
        elif sigma.dim() == 2:
            sigma = sigma[None, :, :]
    return torch.exp(-0.5 * (diff / sigma) ** 2)

def fit_lstsq(B, y):
    in_dim = B.shape[1]
    out_dim = y.shape[-1]
    n_basis = B.shape[-1]
    device = B.device
    coef = torch.zeros(in_dim, out_dim, n_basis, device=device, dtype=B.dtype)
    B_perm = B.permute(1, 0, 2)
    y_perm = y.permute(1, 0, 2)
    for i in range(in_dim):
        sol = torch.linalg.lstsq(B_perm[i], y_perm[i]).solution
        coef[i] = sol.T
    return coef