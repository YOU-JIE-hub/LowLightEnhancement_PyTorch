import torch

def apply_curve(x, A, r=8):
    """
    x: input tensor (B, 3, H, W)
    A: curve tensor (B, 24, H, W)
    r: number of iterations (default 8)
    """
    B, C, H, W = x.size()
    A = A.view(B, r, C, H, W)

    for i in range(r):
        a = A[:, i, :, :, :]
        x = x + a * (torch.pow(x, 2) - x)
    return torch.clamp(x, 0, 1)
