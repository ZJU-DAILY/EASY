import torch
from torch import Tensor


def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)


def view_back(M):
    return view3(M) if M.dim() == 2 else view2(M)


def cosine_sim(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cosine_distance(x1, x2, eps=1e-8):
    return 1 - cosine_sim(x1, x2, eps)
