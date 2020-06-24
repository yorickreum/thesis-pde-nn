import itertools
from collections.abc import Iterable

import torch
from torch.autograd import Variable


def is_iterable(obj):
    return isinstance(obj, Iterable)


def differentiate(fx: torch.Tensor, x: torch.Tensor, n: int):
    """
    n-order differentiation of (evaluated) 1d-function fx w.r.t. x
    :rtype: torch.Tensor
    """
    dfx = fx
    for i in range(n):
        if dfx != 0:
            # @TODO handle if only one element of gradient gets zero
            dfx = torch.autograd.grad(dfx, x, create_graph=True)[0]
        else:
            return dfx
    return dfx


def test_differentiate():
    # 1-dim
    x = torch.tensor(5, dtype=torch.float, requires_grad=True)
    fx = x ** 3 + 5 * x ** 2 + 15
    assert x == 5
    assert fx == 265
    assert differentiate(fx, x, 1) == 125
    assert differentiate(fx, x, 2) == 40
    assert differentiate(fx, x, 3) == 6
    assert differentiate(fx, x, 4) == 0
    assert differentiate(fx, x, 5) == 0


test_differentiate()


def laplace(fx: torch.Tensor, x: torch.Tensor):
    """
    laplace (= sum of 2nd derivations)
     of (evaluated) nd->1d-function fx w.r.t. nd-tensor x
    :rtype: torch.Tensor
    """
    dfx = fx
    dfx = torch.autograd.grad(dfx, x, create_graph=True)[0]
    ddfx = []
    for i in range(len(x)):
        vec = torch.tensor([(1 if i == j else 0) for j in range(len(dfx))], dtype=torch.float)
        ddfx += [torch.autograd.grad(
            dfx,
            x,
            create_graph=True,
            grad_outputs=vec
        )[0][i]]
    ret = sum(ddfx)
    return ret


def test_laplace():
    # 2-dim, test laplace
    x = torch.tensor([5, 4], dtype=torch.float, requires_grad=True)
    fx = 5 * x[0] ** 2 + x[1] ** 5 + 15
    assert fx == 1164
    assert laplace(fx, x) == 1290

    x = torch.tensor([5, 4], dtype=torch.float, requires_grad=True)
    fx = 2 * x[0] ** 4 + x[1] ** 3 + 20
    assert fx == 1334
    assert laplace(fx, x) == 624

    x = torch.tensor([5, 4], dtype=torch.float, requires_grad=True)
    fx = 2 * x[0] ** 4 + x[1] ** 3 + (x[0] ** 3) * (x[1] ** 2) + 20
    assert fx == 3334
    assert laplace(fx, x) == 1354

    x = torch.tensor([5, 4, 2], dtype=torch.float, requires_grad=True)
    fx = 2 * x[0] ** 4 + x[1] ** 3 + 2 * x[2] ** 6 + 20
    assert fx == 1462
    assert laplace(fx, x) == 1584

    x = torch.tensor([5, 4, 2], dtype=torch.float, requires_grad=True)
    fx = 2 * x[0] ** 4 + x[1] ** 3 + 2 * x[2] ** 6 + 5 * x[0] ** 1 * x[1] ** 2 * x[0] ** 3 + 20
    assert fx == 51462
    assert laplace(fx, x) == 31834

    x = torch.tensor([5, 4, 2], dtype=torch.float, requires_grad=True)
    fx = 2 * x[0] ** 4 + x[1] ** 3 + 2 * x[2] ** 6 + 5 * x[0] ** 2 * x[1] ** 3 * x[0] ** 4 + 20
    assert fx == 5001462
    assert laplace(fx, x) == 7876584


test_laplace()
