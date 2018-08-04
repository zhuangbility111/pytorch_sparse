import torch
import torch_scatter


def coalesce(index, value, size, op='add', fill_value=0):
    m, n = size
    row, col = index

    unique, inv = torch.unique(row * n + col, sorted=True, return_inverse=True)

    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(inv.max().item() + 1).scatter_(0, inv, perm)
    index = torch.stack([row[perm], col[perm]], dim=0)

    if value is not None:
        op = getattr(torch_scatter, 'scatter_{}'.format(op))
        value = op(value, inv, 0, None, perm.size(0), fill_value)

    return index, value