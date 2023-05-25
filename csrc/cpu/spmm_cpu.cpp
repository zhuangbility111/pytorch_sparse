#include "spmm_cpu.h"

#include <ATen/Parallel.h>
#include <omp.h>

#include "reducer.h"
#include "utils.h"
#include <chrono> 
#include <algorithm>

#ifdef __ARM_FEATURE_SVE
	#include <arm_sve.h>
	#define VEC_LEN 16
#endif /* __ARM_FEATURE_SVE */

using namespace std::chrono;

// typedef void (*inner_kernel)(int, float *, float *, float *, int);
using Inner_kernel = void(*)(int64_t*, float*, float*, float*,
							 int, int, int, int, int,
							 svbool_t&, svbool_t&, svbool_t&, svbool_t&);

template <int N>
void kernel_1xN(int64_t* col, float* value, float* mat, float* out, 
				int m, int n, int ldb, int start_on_cols, int end_on_cols,
				svbool_t& pg0, svbool_t& pg1, svbool_t& pg2, svbool_t& pg3) {
	svfloat32_t vout0, vout1, vout2, vout3;
	svfloat32_t va;
	svfloat32_t vb0, vb1, vb2, vb3;
	int out_idx = m*ldb + n;
	// load output to SVE register
	/*
	if (N > 0)
		vout0 = svdup_n_f32(0.0);
	if (N > 1)
		vout1 = svdup_n_f32(0.0);
	if (N > 2)
		vout2 = svdup_n_f32(0.0);
	if (N > 3)
		vout3 = svdup_n_f32(0.0);
	*/
	if (N > 0)
		vout0 = svld1(pg0, &(out[out_idx]));
	if (N > 1)
		vout1 = svld1(pg1, &(out[out_idx + VEC_LEN]));
	if (N > 2)
		vout2 = svld1(pg2, &(out[out_idx + 2 * VEC_LEN]));
	if (N > 3)
		vout3 = svld1(pg3, &(out[out_idx + 3 * VEC_LEN]));

				/*
				for (int n = start_on_N; n < end_on_N; n += VEC_LEN) {
					svbool_t pg = svwhilelt_b32(n, end_on_N);
					// svfloat32_t vout = svld1(pg, &(out_data[m*N + n]));
					svfloat32_t vout = svdup_n_f32(0.0);

					for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
						int k = col_data[id_on_cols];
						// float value = value_data[id_on_cols];
						// out_data[m*N + n] += value * mat_data[k*N + n];
						svfloat32_t va = svdup_n_f32(value_data[id_on_cols]);
						svfloat32_t vb = svld1(pg, &(mat_data[k*N + n]));
						vout = svmla_f32_x(pg, vout, va, vb);
					}
					svst1(pg, &(out_data[m*N + n]), vout);
				}
				*/

	for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
		int k = col[id_on_cols];
		int b_idx = k*ldb + n;
		// load elem on sparse matrix
		va = svdup_n_f32(value[id_on_cols]);
		// load elems on dense matrix based on the value of N
		if (N > 0)
			vb0 = svld1(pg0, &(mat[b_idx]));
		if (N > 1)
			vb1 = svld1(pg1, &(mat[b_idx + VEC_LEN]));
		if (N > 2)
			vb2 = svld1(pg2, &(mat[b_idx + 2 * VEC_LEN]));
		if (N > 3)
			vb3 = svld1(pg3, &(mat[b_idx + 3 * VEC_LEN]));

		// fma based on the value of N
		if (N > 0)
			vout0 = svmla_f32_x(pg0, vout0, va, vb0);
		if (N > 1)
			vout1 = svmla_f32_x(pg1, vout1, va, vb1);
		if (N > 2)
			vout2 = svmla_f32_x(pg2, vout2, va, vb2);
		if (N > 3)
			vout3 = svmla_f32_x(pg3, vout3, va, vb3);
	}

	// store output from SVE register
	if (N > 0)
		svst1(pg0, &(out[out_idx]), vout0);
	if (N > 1)
		svst1(pg1, &(out[out_idx + VEC_LEN]), vout1);
	if (N > 2)
		svst1(pg2, &(out[out_idx + 2 * VEC_LEN]), vout2);
	if (N > 3)
		svst1(pg3, &(out[out_idx + 3 * VEC_LEN]), vout3);
}

Inner_kernel get_kernel_1xN(int n) {
	if (n == 1)
		return kernel_1xN<1>;
	else if (n == 2)
		return kernel_1xN<2>;
	else if (n == 3)
		return kernel_1xN<3>;
	return kernel_1xN<4>;
}

inline int divup(int x, int y) {
	return (x + y - 1) / y;	
}

void divide_work(int* work_range, int total_work, int num_threads) {
	int chunk_size;
	int remain_work = total_work;
	work_range[0] = 0;
	for (int i = 0; i < num_threads; i++) {
		chunk_size = divup(remain_work, num_threads - i);
		work_range[i+1] = work_range[i] + chunk_size;
		remain_work -= chunk_size;
	}
	work_range[num_threads] = total_work;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce) {
  auto other_start_time = system_clock::now();
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
 if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());


  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  // duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
	// std::cout << "elapsed time of other part " << "(spmm on forward): " << diff_other.count() << std::endl;	

  auto start_time_1 = system_clock::now();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    scalar_t *value_data = nullptr;
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            b = i / M, m = i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              c = col_data[e];
              if (HAS_VALUE)
                val = value_data[e];
              for (auto k = 0; k < K; k++) {
                if (HAS_VALUE)
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], val * mat_data[offset + c * K + k], &args[k],
                      e);
                else
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], mat_data[offset + c * K + k], &args[k], e);
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });
  // duration<double, std::milli> diff = (system_clock::now() - start_time_1);
  // std::cout << "elapsed time of original kernel:" << diff.count() << std::endl;
  // duration<double, std::milli> diff3 = (system_clock::now() - other_start_time);
  // std::cout << "original total time of SPMM:" << diff3.count() << std::endl;

  return std::make_tuple(out, arg_out);
}

int obtain_tile_rowptr(int64_t* rowptr, int64_t* col, float* values,
						  int rowptr_start, int rowptr_end,
						  int* tile_rowptr, int tile_num, int tile_size) {
	// std::cout << "tile_num: " << tile_num << ", tile_size: " << tile_size << std::endl;
	int tile_rowptr_id = 0;
	int cur_tile_id = 0;
	tile_rowptr[0] = rowptr[rowptr_start];
	for (int i = rowptr_start; i < rowptr_end; i++) {
		int col_id_start = rowptr[i], col_id_end = rowptr[i+1];
		int cur_col_id = col_id_start;
		int cur_col = col[cur_col_id];
		for (int cur_tile_id = 0; cur_tile_id < tile_num; cur_tile_id++) {
			++tile_rowptr_id;
			tile_rowptr[tile_rowptr_id] = tile_rowptr[tile_rowptr_id - 1];
			while (cur_col_id < col_id_end && cur_col / tile_size == cur_tile_id) {
				// std::cout << "row: " << i << ", " << "cur_col: " << cur_col << ", tile_id: " << cur_col / tile_size << ", cur_tile_id: " << cur_tile_id << std::endl;
				tile_rowptr[tile_rowptr_id]++;
				++cur_col_id;
				cur_col = col[cur_col_id];
			}
		}
	}
	return 1;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized(torch::Tensor rowptr, torch::Tensor col, 
				   torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   int64_t sparse_rows, std::string reduce) {

	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);

	mat = mat.contiguous();

	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	auto out = torch::zeros(sizes, mat.options());

	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();

	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int64_t dense_rows = mat.size(-2);
	int64_t dense_cols = mat.size(-1);
	int64_t dense_batch_size = mat.numel() / (dense_rows * dense_cols);
	int64_t sparse_cols = dense_rows;

	int K = static_cast<int>(sparse_cols);
	int N = static_cast<int>(dense_cols);

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {
		float* value_data = nullptr;
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();

		const bool HAS_VALUE = optional_value.has_value();

		if (HAS_VALUE)
			value_data = optional_value.value().data_ptr<float>();

		auto tile_start_time = system_clock::now();

		int64_t tile_num_on_K = 1;
		int64_t tile_num_on_M = 1;
		int64_t tile_size_on_K = divup(sparse_cols, tile_num_on_K);
		int64_t tile_size_on_M = divup(sparse_rows, tile_num_on_M);
		int64_t tile_rowptr_size = tile_size_on_M * tile_num_on_K + 1;
		int tile_rowptr[tile_num_on_M][tile_rowptr_size];

		for (int i = 0; i < tile_num_on_M; i++) {
			obtain_tile_rowptr(rowptr_data, col_data, value_data,
							   i*tile_size_on_M, std::min((i+1)*tile_size_on_M, sparse_rows),
							   tile_rowptr[i], tile_num_on_K, tile_size_on_K);
		}

		auto tile_end_time = system_clock::now();
		duration<double, std::milli> tile_elapsed_time = tile_end_time - tile_start_time;
		std::cout << "tile rowptr time(spmm on forward): " << tile_elapsed_time.count() << "ms" << std::endl;

		// int64_t batch_times_rows = dense_batch_size * sparse_rows;
		int max_num_threads = omp_get_max_threads();
		// std::cout << "max_num_threads = " << max_num_threads << std::endl;
		int num_threads_on_vertexs = max_num_threads;
		int num_threads_on_features = 1;

		int work_range_on_vertexs[num_threads_on_vertexs + 1];
		int work_range_on_features[num_threads_on_features + 1];

		double elapsed_time_array[num_threads_on_vertexs];
		
		auto start_time_1 = system_clock::now();
		// divide work
		divide_work(work_range_on_vertexs, sparse_rows, num_threads_on_vertexs);
		divide_work(work_range_on_features, dense_cols, num_threads_on_features);
		duration<double, std::milli> diff = (system_clock::now() - start_time_1);
		// std::cout << "elapsed time of dividing work(spmm on forward): " << diff.count() << std::endl;

		auto start_time = system_clock::now();
		#pragma omp parallel 
		{
			// auto start_time = system_clock::now();
			int tid = omp_get_thread_num();
			int tid_on_vertexs = tid / num_threads_on_features;
			int tid_on_features = tid % num_threads_on_features;

			int start_on_M = work_range_on_vertexs[tid_on_vertexs];
			int end_on_M = work_range_on_vertexs[tid_on_vertexs + 1];

			int start_on_N = work_range_on_features[tid_on_features];
			int end_on_N = work_range_on_features[tid_on_features + 1];
/*
			std::cout << "tid = " << tid << " tid_on_vertexs = " << tid_on_vertexs
										 << " tid_on_features = " << tid_on_features
										 << " start_on_v = " << work_range_on_vertexs[0]
										 << " end_on_v = " << work_range_on_vertexs[1]
										 << " start_on_f = " << work_range_on_features[0]
										 << " end_on_f = " << work_range_on_features[1]
										 << std::endl;
*/
			// for (int tile_id_on_M = 0; tile_id_on_M < tile_num_on_M; tile_id_on_M++) {	
			// 	start_on_M = tile_id_on_M * tile_size_on_M;
			// 	end_on_M = std::min((tile_id_on_M+1) * tile_size_on_M, sparse_rows);
				int tile_id_on_M = 0;
				for (int tile_id_on_K = 0; tile_id_on_K < tile_num_on_K; tile_id_on_K++) {
					for (int m = start_on_M; m < end_on_M; m++) {
						int idx_on_tile_rowptr = m * tile_num_on_K + tile_id_on_K;
						int start_on_cols = tile_rowptr[tile_id_on_M][idx_on_tile_rowptr];
						int end_on_cols = tile_rowptr[tile_id_on_M][idx_on_tile_rowptr + 1];
						for (int n = start_on_N; n < end_on_N; n += VEC_LEN) {
							svbool_t pg = svwhilelt_b32(n, end_on_N);
							svfloat32_t vout = svld1(pg, &(out_data[m*N + n]));
							for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
								int k = col_data[id_on_cols];
								// float value = value_data[id_on_cols];
								// std::cout << "m = " << m << ", k = " << k << ", value = " << value << std::endl;
								// out_data[m*N + n] += value * mat_data[k*N + n];
								svfloat32_t va = svdup_n_f32(value_data[id_on_cols]);
								svfloat32_t vb = svld1(pg, &(mat_data[k*N + n]));
								vout = svmla_f32_x(pg, vout, va, vb);
							}
							svst1(pg, &(out_data[m*N + n]), vout);
						}
					}
				}
				/*
				duration<double, std::milli> diff1 = (system_clock::now() - start_time);
				elapsed_time_array[tid] = diff1.count();
				*/
			// }
		}
		duration<double, std::milli> diff1 = (system_clock::now() - start_time);
		std::cout << "elapsed time of kernel " << "(spmm on forward): " << diff1.count() << std::endl;	
/*
		for (int i = 0; i < num_threads_on_vertexs; i++) {
			std::cout << "elapsed time of thread " << i << "(spmm on forward): " << elapsed_time_array[i] << std::endl;	
		}
*/
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized_no_tile(torch::Tensor rowptr, torch::Tensor col, 
				   		   torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
				   		   int64_t sparse_rows, std::string reduce) {

	// auto other_start_time = system_clock::now();
	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);

	mat = mat.contiguous();

	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	// auto out = torch::zeros(sizes, mat.options());
	
	// auto other_start_time = system_clock::now();
	// use empty to accelerate the process of allocating memory 
	auto out = torch::empty(sizes, mat.options());
	// duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
	// std::cout << "elapsed time of allocating memory " << "(spmm on forward): " << diff_other.count() << std::endl;	

	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();

	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int K = mat.size(-2);
	int N = mat.size(-1);
	int M = (int)(sparse_rows);
	int dense_batch_size = (int)(mat.numel()) / (K * N);
	// int64_t sparse_cols = dense_rows;

/*
	int K = static_cast<int>(sparse_cols);
	int N = static_cast<int>(dense_cols);
*/

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {
		float* value_data = nullptr;
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();

		const bool HAS_VALUE = optional_value.has_value();

		if (HAS_VALUE)
			value_data = optional_value.value().data_ptr<float>();

		// int64_t batch_times_rows = dense_batch_size * sparse_rows;
		int max_num_threads = omp_get_max_threads();
		// std::cout << "max_num_threads = " << max_num_threads << std::endl;
		int num_threads_on_vertexs = max_num_threads;
		int num_threads_on_features = 1;

		int work_range_on_vertexs[num_threads_on_vertexs + 1];
		int work_range_on_features[num_threads_on_features + 1];
		// int* work_range_on_vertexs = (int*)malloc(sizeof(int) * (num_threads_on_vertexs + 1));
		// int* work_range_on_features = (int*)malloc(sizeof(int) * (num_threads_on_features + 1));

		// double elapsed_time_array[num_threads_on_vertexs];
		
	 	// auto start_time_1 = system_clock::now();
		// divide work
		divide_work(work_range_on_vertexs, M, num_threads_on_vertexs);
		divide_work(work_range_on_features, N, num_threads_on_features);
		// duration<double, std::milli> diff = (system_clock::now() - start_time_1);
		// std::cout << "elapsed time of dividing work(spmm on forward): " << diff.count() << std::endl;
		// duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
		// std::cout << "elapsed time of other part " << "(spmm on forward): " << diff_other.count() << std::endl;	
	// duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
	// std::cout << "elapsed time of other part " << "(spmm on forward): " << diff_other.count() << std::endl;	

		auto start_time = system_clock::now();
		#pragma omp parallel 
		{
			// auto start_time = system_clock::now();
			int tid = omp_get_thread_num();
			int tid_on_vertexs = tid / num_threads_on_features;
			int tid_on_features = tid % num_threads_on_features;

			int start_on_M = work_range_on_vertexs[tid_on_vertexs];
			int end_on_M = work_range_on_vertexs[tid_on_vertexs + 1];

			int start_on_N = work_range_on_features[tid_on_features];
			int end_on_N = work_range_on_features[tid_on_features + 1];

			for (int m = start_on_M; m < end_on_M; m++) {
				int start_on_cols = rowptr_data[m];
				int end_on_cols = rowptr_data[m+1];
				for (int n = start_on_N; n < end_on_N; n += VEC_LEN) {
					svbool_t pg = svwhilelt_b32(n, end_on_N);
					// svfloat32_t vout = svld1(pg, &(out_data[m*N + n]));
					svfloat32_t vout = svdup_n_f32(0.0);

					for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
						int k = col_data[id_on_cols];
						// float value = value_data[id_on_cols];
						// out_data[m*N + n] += value * mat_data[k*N + n];
						svfloat32_t va = svdup_n_f32(value_data[id_on_cols]);
						svfloat32_t vb = svld1(pg, &(mat_data[k*N + n]));
						vout = svmla_f32_x(pg, vout, va, vb);
					}
					svst1(pg, &(out_data[m*N + n]), vout);
				}
			}
				/*
				duration<double, std::milli> diff1 = (system_clock::now() - start_time);
				elapsed_time_array[tid] = diff1.count();
				*/
		}
		duration<double, std::milli> diff1 = (system_clock::now() - start_time);
		std::cout << "elapsed time of no tile's kernel = " << diff1.count() << "ms" << std::endl;	
/*
		for (int i = 0; i < num_threads_on_vertexs; i++) {
			std::cout << "elapsed time of thread " << i << "(spmm on forward): " << elapsed_time_array[i] << std::endl;	
		}
*/
	
		// free(work_range_on_vertexs);
		// free(work_range_on_features);
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}

Inner_kernel select_kernel(const int N, int& step, 
						   svbool_t& pg0, svbool_t& pg1, svbool_t& pg2, svbool_t& pg3) {
	Inner_kernel kernel = nullptr;
	if (N > 3 * VEC_LEN) {
		kernel = get_kernel_1xN(4);
		pg3 = svwhilelt_b32(3 * VEC_LEN, N);
		step = std::min(N, 4 * VEC_LEN);
	} else if (N > 2 * VEC_LEN) {
		kernel = get_kernel_1xN(3);
		pg2 = svwhilelt_b32(2 * VEC_LEN, N);
		step = N;
	} else if (N > 1 * VEC_LEN) {
		kernel = get_kernel_1xN(2);
		pg1 = svwhilelt_b32(1 * VEC_LEN, N);
		step = N;
	} else if (N > 0) {
		kernel = get_kernel_1xN(1);
		pg0 = svwhilelt_b32(0 * VEC_LEN, N);
		step = N;
	}
	return kernel;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_optimized_no_tile_v1(torch::Tensor rowptr, torch::Tensor col, 
				   		   	  torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
							  torch::Tensor out,
				   		   	  int64_t sparse_rows, std::string reduce) {

	// auto other_start_time = system_clock::now();
	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);

	mat = mat.contiguous();

	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	// auto out = torch::zeros(sizes, mat.options());
	
	auto other_start_time = system_clock::now();
	// use empty to accelerate the process of allocating memory 
	// auto out = torch::empty(sizes, mat.options());
	duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
	std::cout << "elapsed time of allocating memory " << "(spmm on forward): " << diff_other.count() << std::endl;	


	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();

	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int K = mat.size(-2);
	int N = mat.size(-1);
	int M = (int)(sparse_rows);
	int dense_batch_size = (int)(mat.numel()) / (K * N);
	// int64_t sparse_cols = dense_rows;

/*
	int K = static_cast<int>(sparse_cols);
	int N = static_cast<int>(dense_cols);
*/

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {
		float* value_data = nullptr;
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();

		const bool HAS_VALUE = optional_value.has_value();

		if (HAS_VALUE)
			value_data = optional_value.value().data_ptr<float>();

		// int64_t batch_times_rows = dense_batch_size * sparse_rows;
		int max_num_threads = omp_get_max_threads();
		// std::cout << "max_num_threads = " << max_num_threads << std::endl;
		int num_threads_on_vertexs = max_num_threads;
		int num_threads_on_features = 1;

		int work_range_on_vertexs[num_threads_on_vertexs + 1];
		int work_range_on_features[num_threads_on_features + 1];
		// int* work_range_on_vertexs = (int*)malloc(sizeof(int) * (num_threads_on_vertexs + 1));
		// int* work_range_on_features = (int*)malloc(sizeof(int) * (num_threads_on_features + 1));

		// double elapsed_time_array[num_threads_on_vertexs];
		
	 	// auto start_time_1 = system_clock::now();
		// divide work
		divide_work(work_range_on_vertexs, M, num_threads_on_vertexs);
		divide_work(work_range_on_features, N, num_threads_on_features);
		// duration<double, std::milli> diff = (system_clock::now() - start_time_1);
		// std::cout << "elapsed time of dividing work(spmm on forward): " << diff.count() << std::endl;
		// duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
		// std::cout << "elapsed time of other part " << "(spmm on forward): " << diff_other.count() << std::endl;	
	// duration<double, std::milli> diff_other = (system_clock::now() - other_start_time);
	// std::cout << "elapsed time of other part " << "(spmm on forward): " << diff_other.count() << std::endl;	

		auto start_time = system_clock::now();
		#pragma omp parallel 
		{
			// auto start_time = system_clock::now();
			int tid = omp_get_thread_num();
			int tid_on_vertexs = tid / num_threads_on_features;
			int tid_on_features = tid % num_threads_on_features;

			int start_on_M = work_range_on_vertexs[tid_on_vertexs];
			int end_on_M = work_range_on_vertexs[tid_on_vertexs + 1];

			int start_on_N = work_range_on_features[tid_on_features];
			int end_on_N = work_range_on_features[tid_on_features + 1];
			int step_on_N = end_on_N - start_on_N;

			svbool_t pg0_main = svptrue_b32();
			svbool_t pg1_main = svptrue_b32();
			svbool_t pg2_main = svptrue_b32();
			svbool_t pg3_main = svptrue_b32();

			svbool_t pg0_corner = svptrue_b32();
			svbool_t pg1_corner = svptrue_b32();
			svbool_t pg2_corner = svptrue_b32();
			svbool_t pg3_corner = svptrue_b32();

			// select kernel
			int step_main_kernel = 0, step_corner_kernel = 0;
			Inner_kernel main_kernel = select_kernel(step_on_N, step_main_kernel, pg0_main, pg1_main, pg2_main, pg3_main);
			Inner_kernel corner_kernel = select_kernel(step_on_N % step_main_kernel, step_corner_kernel, pg0_corner, pg1_corner, pg2_corner, pg3_corner);
			int start_on_N_main = start_on_N;
			int end_on_N_main = end_on_N - step_corner_kernel;
			int start_on_N_corner = end_on_N_main;
			int end_on_N_corner = end_on_N;

			/*
			printf("step_main_kernel = %d, step_corner_kernel = %d\n", step_main_kernel, step_corner_kernel);
			printf("start_on_N_main = %d, end_on_N_main = %d\n", start_on_N_main, end_on_N_main);
			printf("start_on_N_corner = %d, end_on_N_corner = %d\n", start_on_N_corner, end_on_N_corner);
			*/

			for (int m = start_on_M; m < end_on_M; m++) {
				int start_on_cols = rowptr_data[m];
				int end_on_cols = rowptr_data[m+1];
				for (int n = start_on_N_main; n < end_on_N_main; n += step_main_kernel) {
					main_kernel(col_data, value_data, mat_data, out_data,
								m, n, N, start_on_cols, end_on_cols,
								pg0_main, pg1_main, pg2_main, pg3_main);	
				}

				if (start_on_N_corner < end_on_N_corner) {
					corner_kernel(col_data, value_data, mat_data, out_data,
								  m, start_on_N_corner, N, start_on_cols, end_on_cols,
								  pg0_corner, pg1_corner, pg2_corner, pg3_corner);
				}
				/*
				for (int n = start_on_N; n < end_on_N; n += VEC_LEN) {
					svbool_t pg = svwhilelt_b32(n, end_on_N);
					// svfloat32_t vout = svld1(pg, &(out_data[m*N + n]));
					svfloat32_t vout = svdup_n_f32(0.0);

					for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
						int k = col_data[id_on_cols];
						// float value = value_data[id_on_cols];
						// out_data[m*N + n] += value * mat_data[k*N + n];
						svfloat32_t va = svdup_n_f32(value_data[id_on_cols]);
						svfloat32_t vb = svld1(pg, &(mat_data[k*N + n]));
						vout = svmla_f32_x(pg, vout, va, vb);
					}
					svst1(pg, &(out_data[m*N + n]), vout);
				}
				*/
			}
				/*
				duration<double, std::milli> diff1 = (system_clock::now() - start_time);
				elapsed_time_array[tid] = diff1.count();
				*/
		}
		duration<double, std::milli> diff1 = (system_clock::now() - start_time);
		std::cout << "elapsed time of no tile's kernel = " << diff1.count() << "ms" << std::endl;	
  		duration<double, std::milli> diff3 = (system_clock::now() - other_start_time);
  		std::cout << "elapsed time of no tile's spmm:" << diff3.count() << std::endl;
/*
		for (int i = 0; i < num_threads_on_vertexs; i++) {
			std::cout << "elapsed time of thread " << i << "(spmm on forward): " << elapsed_time_array[i] << std::endl;	
		}
*/
	
		// free(work_range_on_vertexs);
		// free(work_range_on_features);
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_for_transposed_sparse(torch::Tensor rowptr, torch::Tensor col, 
								torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
								int64_t sparse_rows, std::string reduce) {
	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);

	mat = mat.contiguous();

	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	auto out = torch::zeros(sizes, mat.options());

	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();

	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int64_t sparse_cols = rowptr.numel() - 1;
	int64_t dense_rows = mat.size(-2);
	int64_t dense_cols = mat.size(-1);
	int64_t dense_batch_size = mat.numel() / (dense_rows * dense_cols);

	int64_t K = sparse_cols;
	int64_t N = dense_cols;

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {
		float* value_data = nullptr;
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();

		const bool HAS_VALUE = optional_value.has_value();

		if (HAS_VALUE)
			value_data = optional_value.value().data_ptr<float>();

		int64_t batch_times_rows = dense_batch_size * sparse_rows;
		for (int64_t k = 0; k < K; k++) {
			int64_t row_start = rowptr_data[k], row_end = rowptr_data[k + 1];

			// #pragma omp parallel for 
			for (int64_t idx_in_col_data = row_start; idx_in_col_data < row_end; idx_in_col_data++) {
				int64_t m = col_data[idx_in_col_data];
				float value = value_data[idx_in_col_data];

				for (int64_t n = 0; n < N; n++) {
					out_data[m*N + n] += value * mat_data[k*N + n];
				}
			}
		}
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}


std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu_for_transposed_sparse_multi_threads(torch::Tensor rowptr, torch::Tensor col, 
										     torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
											 int64_t sparse_rows, std::string reduce) {
	// check sparse matrix ptr
	CHECK_CPU(rowptr);
	CHECK_CPU(col);
	if (optional_value.has_value())
		CHECK_CPU(optional_value.value());
	CHECK_CPU(mat);
	
	// check shape of sparse matrix 
	CHECK_INPUT(rowptr.dim() == 1);
	CHECK_INPUT(col.dim() == 1);

	if (optional_value.has_value()) {
		CHECK_INPUT(optional_value.value().dim() == 1);
		CHECK_INPUT(optional_value.value().size(0) == col.size(0));
	}
	CHECK_INPUT(mat.dim() == 2);

	mat = mat.contiguous();

	// allocate output memory
	auto sizes = mat.sizes().vec();
	sizes[mat.dim() - 2] = sparse_rows;
	auto out = torch::zeros(sizes, mat.options());

	auto rowptr_data = rowptr.data_ptr<int64_t>();
	auto col_data = col.data_ptr<int64_t>();

	// sparse matrix shape: M * K
	// dense matrix shape: B * K * N
	int64_t sparse_cols = rowptr.numel() - 1;
	int64_t dense_rows = mat.size(-2);
	int64_t dense_cols = mat.size(-1);
	int64_t dense_batch_size = mat.numel() / (dense_rows * dense_cols);

	int K = static_cast<int>(sparse_cols);
	int N = static_cast<int>(dense_cols);

	if (mat.scalar_type() == at::ScalarType::Float && 
		optional_value.value().scalar_type() == at::ScalarType::Float) {
		float* value_data = nullptr;
		float* mat_data = mat.data_ptr<float>();
		float* out_data = out.data_ptr<float>();

		const bool HAS_VALUE = optional_value.has_value();

		if (HAS_VALUE)
			value_data = optional_value.value().data_ptr<float>();

		// int64_t batch_times_rows = dense_batch_size * sparse_rows;
		int max_num_threads = omp_get_max_threads();
		// std::cout << "max_num_threads = " << max_num_threads << std::endl;
		int num_threads_on_vertexs = max_num_threads;
		int num_threads_on_features = 1;

		auto tile_start_time = system_clock::now();

		int64_t tile_num = num_threads_on_vertexs;
		int64_t tile_size = divup(sparse_rows, tile_num);
		int64_t tile_rowptr_size = sparse_cols * tile_num + 1;
		int tile_rowptr[tile_rowptr_size];
		obtain_tile_rowptr(rowptr_data, col_data, value_data,
						   0, sparse_cols,
						   tile_rowptr, tile_num, tile_size);

		auto tile_end_time = system_clock::now();
		duration<double, std::milli> tile_elapsed_time = tile_end_time - tile_start_time;
		std::cout << "tile rowptr time: " << tile_elapsed_time.count() << "ms" << std::endl;

		int work_range_on_vertexs[num_threads_on_vertexs + 1];
		int work_range_on_features[num_threads_on_features + 1];

		double elapsed_time_array[num_threads_on_vertexs];
		
		auto start_time_1 = system_clock::now();
		// divide work
		divide_work(work_range_on_vertexs, sparse_rows, num_threads_on_vertexs);
		divide_work(work_range_on_features, dense_cols, num_threads_on_features);
		duration<double, std::milli> diff = (system_clock::now() - start_time_1);
		// std::cout << "elapsed time of dividing work: " << diff.count() << std::endl;

		auto start_time = system_clock::now();
		#pragma omp parallel 
		{
			// auto start_time = system_clock::now();
			int tid = omp_get_thread_num();
			int tid_on_vertexs = tid / num_threads_on_features;
			int tid_on_features = tid % num_threads_on_features;

/*
			int start_on_vertexs = work_range_on_vertexs[tid_on_vertexs];
			int end_on_vertexs = work_range_on_vertexs[tid_on_vertexs + 1];
*/

			int start_on_features = work_range_on_features[tid_on_features];
			int end_on_feautures = work_range_on_features[tid_on_features + 1];
/*
			std::cout << "tid = " << tid << " tid_on_vertexs = " << tid_on_vertexs
										 << " tid_on_features = " << tid_on_features
										 << " start_on_v = " << work_range_on_vertexs[0]
										 << " end_on_v = " << work_range_on_vertexs[1]
										 << " start_on_f = " << work_range_on_features[0]
										 << " end_on_f = " << work_range_on_features[1]
										 << std::endl;
*/
			for (int64_t k = 0; k < K; k++) {
				int idx_on_tile_rowptr = k * tile_num + tid_on_vertexs;
				int64_t row_start = tile_rowptr[idx_on_tile_rowptr];
				int64_t row_end = tile_rowptr[idx_on_tile_rowptr + 1];

				// #pragma omp parallel for 
				for (int64_t idx_in_col_data = row_start; idx_in_col_data < row_end; idx_in_col_data++) {
					int64_t m = col_data[idx_in_col_data];
					float value = value_data[idx_in_col_data];

					for (int64_t n = 0; n < N; n++) {
						out_data[m*N + n] += value * mat_data[k*N + n];
					}
				}
			}
		/*	
			for (int k = 0; k < K; k++) {
				int row_start = rowptr_data[k];
				int row_end = rowptr_data[k + 1];

				for (int idx_in_col_data = row_start; idx_in_col_data < row_end; idx_in_col_data++) {
					int m = col_data[idx_in_col_data];
					if (m >= start_on_vertexs && m < end_on_vertexs) {
						float value = value_data[idx_in_col_data];

						for (int n = start_on_features; n < end_on_feautures; n++) {
							out_data[m*N + n] += value * mat_data[k*N + n];
						}
					}
				}
			}
			duration<double, std::milli> diff1 = (system_clock::now() - start_time);
			elapsed_time_array[tid] = diff1.count();
		*/
		}
		duration<double, std::milli> diff1 = (system_clock::now() - start_time);
		std::cout << "elapsed time of kernel " << diff1.count() << std::endl;	

/*
		for (int i = 0; i < num_threads_on_vertexs; i++) {
			std::cout << "elapsed time of thread " << i << ": " << elapsed_time_array[i] << std::endl;	
		}
*/
	}
	else {
		std::cout << "the data type of one input matrix is not float" 
				  << ", float type: " << at::ScalarType::Float
				  << ", sparse_data_type: " << optional_value.value().scalar_type()
				  << ", dense_data_type: " << mat.scalar_type()
				  << std::endl;
	}

	torch::optional<torch::Tensor> arg_out = torch::nullopt;
	return std::make_tuple(out, arg_out);
}

torch::Tensor spmm_value_bw_cpu(torch::Tensor row, torch::Tensor rowptr,
                                torch::Tensor col, torch::Tensor mat,
                                torch::Tensor grad, std::string reduce) {
  CHECK_CPU(row);
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(mat);
  CHECK_CPU(grad);

  mat = mat.contiguous();
  grad = grad.contiguous();

  auto M = grad.size(-2);
  auto N = mat.size(-2);
  auto E = row.numel();
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  auto out = torch::zeros(row.numel(), grad.options());

  auto row_data = row.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    scalar_t val;
    int64_t row, col;
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (int b = 0; b < B; b++) {
        for (int e = 0; e < E; e++) {
          row = row_data[e], col = col_data[e], val = (scalar_t)0;
          for (int k = 0; k < K; k++) {
            val += mat_data[b * N * K + col * K + k] *
                   grad_data[b * M * K + row * K + k];
          }
          if (REDUCE == MEAN) {
            int row_start = rowptr_data[row], row_end = rowptr_data[row + 1];
            val /= (scalar_t)std::max(row_end - row_start, 1);
          }
          out_data[e] += val;
        }
      }
    });
  });

  return out;
}
