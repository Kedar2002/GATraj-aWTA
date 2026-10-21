"""Annealed regression assignment adapted from Xu et al., ICRA 2025.
Independent implementation of the published detached softmin rule.
"""
import math
import torch


def temperature(epoch, total_epochs, initial=0.5, final=0.01, fraction=0.3):
    """Absolute-epoch schedule in metres; hard WTA after the annealing phase."""
    if not (0 < final <= initial and 0 < fraction <= 1 and total_epochs >= 1):
        raise ValueError('Require 0 < final <= initial, 0 < fraction <= 1, total_epochs >= 1')
    end = max(1, round(total_epochs * fraction))
    if epoch > end:
        return 0.0
    progress = max(0, epoch - 1) / max(1, end - 1)
    return initial * math.exp(math.log(final / initial) * progress)


def assignment_weights(ade, tau):
    """ade is [modes, agents]. Assignment itself never receives gradients."""
    if tau <= 0:
        winners = ade.detach().argmin(dim=0, keepdim=True)
        return torch.zeros_like(ade).scatter_(0, winners, 1.0)
    return torch.softmax(-ade.detach() / tau, dim=0)


def annealed_regression(predictions, target, ade, tau, gaussian=False, eps=1e-6):
    """Weighted original per-coordinate NLL, reduced over time then agents."""
    loc, scale = predictions.chunk(2, dim=-1)
    # Match GATraj's clamped forward value and straight-through derivative.
    scale = scale + (scale.clamp_min(eps) - scale).detach()
    if gaussian:
        nll = 0.5 * (torch.log(scale.square()) + (target - loc).square() / scale.square())
    else:
        nll = torch.log(2 * scale) + (target - loc).abs() / scale
    per_mode = nll.mean(dim=(-1, -2))
    return (assignment_weights(ade, tau) * per_mode).sum(dim=0).mean()
