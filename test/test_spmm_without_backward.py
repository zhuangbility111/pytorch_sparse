import torch
import numpy as np
import torch_scatter
from torch_sparse.matmul import matmul, spmm_sum_without_backward
from torch_sparse.tensor import SparseTensor
import scipy.sparse as sci_sp
import time

def check_error(res_ref, res, rows, cols, error):
    total_diff = 0.0
    for i in range(rows):
        for j in range(cols):
            diff = abs(res_ref[i][j] - res[i][j])
            if diff > error:
                print('row = {}, col = {}, ref = {}, res = {}, err = {}'.format(i, j, res_ref[i][j], res[i][j], diff))
                total_diff += diff

    return total_diff

def generate_sparse_tensor(row, col, density, format):
    sci_sparse_mat = sci_sp.rand(row, col, density=density, format=format, dtype=np.float32)
    return SparseTensor.from_scipy(sci_sparse_mat) 

def SPMM_forward(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()
    if value is not None:
        value = value.to(other.dtype)
    return spmm_sum_without_backward(rowptr, col, value, other)

def SPMM_backward(src: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row = src.storage.row()
    csr2csc = src.storage.csr2csc()
    colptr = src.storage.colptr()
    opt_value = value.view(-1, 1).index_select(0, csr2csc).view(-1)
    return spmm_sum_without_backward(colptr, row.index_select(0, csr2csc), opt_value, other)

def test_spmm(dtype, device, reduce):
    '''
    src = torch.randn((10, 8), dtype=dtype, device=device)
    src[2:4, :] = 0  # Remove multiple rows.
    src[:, 2:4] = 0  # Remove multiple columns.
    src = SparseTensor.from_dense(src).requires_grad_()
    row, col, value = src.coo()

    # other = torch.randn((2, 8, 2), dtype=dtype, device=device,
    #                     requires_grad=True)
    other = torch.randn((8, 2), dtype=dtype, device=device,
                        requires_grad=True)
    '''
    density = 0.007
    sparse_format = 'csr'
    sparse_rows = 10000
    sparse_cols = 1000
    dense_rows = sparse_cols
    dense_cols = 333

    src = generate_sparse_tensor(sparse_rows, sparse_cols, density, sparse_format)
    other = torch.rand(dense_rows, dense_cols)
    rowptr, col, value = src.csr()

    '''
    src_col = other.index_select(-2, col) * value.unsqueeze(-1)
    expected = torch_scatter.scatter(src_col, row, dim=-2, reduce=reduce)
    if reduce == 'min':
        expected[expected > 1000] = 0
    if reduce == 'max':
        expected[expected < -1000] = 0

    grad_out = torch.randn_like(expected)
    '''

    # expected.backward(grad_out)
    # expected_grad_value = value.grad
    # value.grad = None
    # expected_grad_other = other.grad
    # other.grad = None

    expected = matmul(src, other, reduce)
    # out.backward(grad_out)
    out = SPMM_forward(src, other)
    # other_grad = SPMM_backward(src, grad_out)

    # check_error(expected, out, sparse_rows, dense_cols, 0.001)

    assert torch.allclose(expected, out, atol=1e-2)
    # assert torch.allclose(expected_grad_value, value.grad, atol=1e-2)
    # assert torch.allclose(expected_grad_other, other.grad, atol=1e-2)
    # assert torch.allclose(expected_grad_other, other_grad, atol=1e-2)

    repeat = 10
    begin = time.perf_counter()
    for _ in range(repeat):
        expected = matmul(src, other, reduce)
    end = time.perf_counter()
    print("elapsed time of original method on {} runs(ms): {}".format(repeat, (end-begin)*1000.0))

    begin = time.perf_counter()
    for _ in range(repeat):
        out = SPMM_forward(src, other)
    end = time.perf_counter()
    print("elapsed time of my method on {} runs(ms): {}".format(repeat, (end-begin)*1000.0))

if __name__ == '__main__':
    test_spmm(torch.float, "cpu", "sum")
