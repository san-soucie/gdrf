import torch


def validate_dirichlet_param(b: torch.Tensor, K: int, V: int, device: str = 'cuda') -> torch.Tensor:
    assert (b <= 0).sum().detach().cpu().item() == 0, "b must be positive"
    if len(b.shape) == 0:
        assert b > 0, "b must be positive if a scalar"
        return torch.ones(K, V).to(device) * b
    elif len(b.shape) == 1:
        if b.shape[0] == K:
            return b.repeat(V, 1).T.to(device)
        elif b.shape[0] == V:
            return b.repeat(K, 1).to(device)
        else:
            raise ValueError("parameter b must have length K or V if 1D")
    elif len(b.shape) == 2:
        assert b.shape == torch.Size((K, V)), "b should be KxV if 2D"
        return b.to(device)
    else:
        raise ValueError("invalid b parameter- you passed %s", b)

def jittercholesky(Kff, N, jitter, maxjitter):
    njitter = 0
    Lff = None
    while njitter < maxjitter:
        try:
            Kff.view(-1)[:: N + 1] += jitter * (10 ** njitter)
            Lff = torch.linalg.cholesky(Kff)
            break
        except RuntimeError:
            njitter += 1
    if njitter >= maxjitter:
        raise RuntimeError("reached max jitter, covariance is unstable")
    return Lff
