import torch
import numpy as np
import torch_scatter
from torch_scatter import segment_csr
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
    return SparseTensor.from_scipy(sci_sparse_mat).requires_grad_()

def distribute_work(rowptr: torch.Tensor, value: torch.Tensor, other: torch.Tensor, 
                    num_threads_on_row: int, num_threads_on_col: int) -> tuple:
    flops_per_row = segment_csr(value, rowptr, None, "sum")
    num_rows = flops_per_row.shape[0]
    num_cols = other.shape[-1]

    total_flops_on_rows = flops_per_row.sum()
    total_flops_on_cols = num_cols

    target_flops_on_rows = total_flops_on_rows // num_threads_on_row
    target_flops_on_cols = total_flops_on_cols // num_threads_on_col

    row_splits = torch.full((num_threads_on_row + 1,), num_rows, dtype=torch.int32)
    col_splits = torch.full((num_threads_on_col + 1,), num_cols, dtype=torch.int32)
    row_splits[0] = 0
    col_splits[0] = 0

    # use greedy algorithm to distribute work for row_splits
    cur_flops_sum = 0
    cur_tid = 0
    for i in range(flops_per_row.shape[0]):
        cur_flops_sum += flops_per_row[i]
        if cur_flops_sum > target_flops_on_rows and cur_tid < num_threads_on_row:
            row_splits[cur_tid+1] = i
            cur_tid += 1
            cur_flops_sum = flops_per_row[i]
    row_splits[-1] = num_rows

    print(row_splits)

    # to distributed work evenly for col_splits
    for i in range(num_threads_on_col):
        col_splits[i] = i * target_flops_on_cols
    col_splits[-1] = num_cols

    # set the work range of remaining threads
    return row_splits, col_splits
    
def SPMM_forward(src: torch.Tensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()

    row_splits, col_splits = src.get_work_range()
    # if row_splits is None and col_splits is None:
    if row_splits.size(0) == 0 and col_splits.size(0) == 0:
        num_threads = torch.get_num_threads()
        row_splits, col_splits = distribute_work(rowptr, value, other, num_threads, 1)
        src.set_work_range(row_splits, col_splits)
    
    if value is not None:
        value = value.to(other.dtype)
    return spmm_sum_without_backward(rowptr, col, value, other, out, row_splits, col_splits)

def SPMM_backward(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # rowptr, col, value = src.csr()
    # row = src.storage.row()
    # csr2csc = src.storage.csr2csc()
    colptr = src.storage.colptr()
    # opt_value = value.view(-1, 1).index_select(0, csr2csc).view(-1)
    row_T = src.storage.row_T()
    value_T = src.storage.value_T()

    row_splits, col_splits = src.get_work_range_for_transpose()
    if row_splits.size(0) == 0 and col_splits.size(0) == 0:
        num_threads = torch.get_num_threads()
        row_splits, col_splits = distribute_work(colptr, value_T, other, num_threads, 1)
        src.set_work_range_for_transpose(row_splits, col_splits)

    # return spmm_sum_without_backward(colptr, row.index_select(0, csr2csc), opt_value, other)
    return spmm_sum_without_backward(colptr, row_T, value_T, other, out, row_splits, col_splits)

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
    sparse_rows = 2002
    sparse_cols = 102
    dense_rows = sparse_cols
    dense_cols = 111

    src = generate_sparse_tensor(sparse_rows, sparse_cols, density, sparse_format)
    other = torch.rand(dense_rows, dense_cols).requires_grad_()
    rowptr, col, value = src.csr()

    '''
    src_col = other.index_select(-2, col) * value.unsqueeze(-1)
    expected = torch_scatter.scatter(src_col, row, dim=-2, reduce=reduce)
    if reduce == 'min':
        expected[expected > 1000] = 0
    if reduce == 'max':
        expected[expected < -1000] = 0

    '''

    expected = matmul(src, other, reduce)
    out = torch.zeros((sparse_rows, dense_cols), dtype=torch.float32)
    SPMM_forward(src, other, out)
    print("expected[0][0:5] = {}".format(expected[0][0:5]))
    print("my res[0][0:5] = {}".format(out[0][0:5]))

    grad_out = torch.randn_like(expected)
    expected.backward(grad_out)
    # expected_grad_value = value.grad
    # value.grad = None
    expected_grad_other = other.grad
    other.grad = None

    other_grad = torch.zeros((sparse_cols, dense_cols), dtype=torch.float32)
    SPMM_backward(src, grad_out, other_grad)
    print("expected grad[0][0:5] = {}".format(expected_grad_other[0][0:5]))
    print("my grad[0][0:5] = {}".format(other_grad[0][0:5]))

    # check_error(expected, out, sparse_rows, dense_cols, 0.001)

    assert torch.allclose(expected, out, atol=1e-5)
    # assert torch.allclose(expected_grad_value, , atol=1e-10)
    assert torch.allclose(expected_grad_other, other_grad, atol=1e-5)
    # assert torch.allclose(expected_grad_other, other_grad, atol=1e-2)

    repeat = 100
    begin = time.perf_counter()
    for _ in range(repeat):
        SPMM_forward(src, other, out)
    end = time.perf_counter()
    print("elapsed time of my method on {} runs(ms): {}, average time(ms): {}".format(repeat, (end-begin)*1000.0, (end-begin)*1000.0/repeat))

    begin = time.perf_counter()
    for _ in range(repeat):
        expected = matmul(src, other, reduce)
    end = time.perf_counter()
    print("elapsed time of original method on {} runs(ms): {}, average time(ms): {}".format(repeat, (end-begin)*1000.0, (end-begin)*1000.0/repeat))

if __name__ == '__main__':
    test_spmm(torch.float, "cpu", "sum")
