#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized(torch::Tensor rowptr, torch::Tensor col, 
				   torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   int64_t sparse_rows, std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized_no_tile(torch::Tensor rowptr, torch::Tensor col, 
				   		   torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   		   int64_t sparse_rows, std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized_no_tile_v1(torch::Tensor rowptr, torch::Tensor col, 
				   			  torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
							  torch::Tensor out,
				   			  int64_t sparse_rows, std::string reduce,
							  torch::Tensor parallel_row_splits, torch::Tensor parallel_col_splits);

torch::Tensor spmm_value_bw_cpu(torch::Tensor row, torch::Tensor rowptr,
                                torch::Tensor col, torch::Tensor mat,
                                torch::Tensor grad, std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_for_transposed_sparse(torch::Tensor rowptr, torch::Tensor col, 
								torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
								int64_t sparse_rows, std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_for_transposed_sparse_multi_threads(torch::Tensor rowptr, torch::Tensor col, 
											 torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
											 int64_t sparse_rows, std::string reduce);

