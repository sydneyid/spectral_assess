import torch
import numpy as np

def sample_mp_singular_values(n, m, sigma=1.0):
    q = n / m if n <= m else m / n
    a = sigma * (1 - np.sqrt(q))**2
    b = sigma * (1 + np.sqrt(q))**2
    svals = np.sqrt(np.random.uniform(a, b, size=min(n, m)))
    return svals

def mp_init_weight(tensor, sigma=1.0):
    n, m = tensor.shape
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(m, m))
    svals = sample_mp_singular_values(n, m, sigma)
    S = torch.zeros(n, m)
    for i in range(len(svals)):
        S[i, i] = svals[i]
    weight = U @ S @ V.T
    with torch.no_grad():
        tensor.copy_(weight)
    return tensor
