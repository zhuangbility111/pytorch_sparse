#include "./spmm_kernel_cpu.h"


template <int N>
void kernel_1xN(float* col, float* value, float* mat, float* out, 
				int m, int n, int N, int start_on_cols, int end_on_cols,
				svbool_t& pg0, svbool_t& pg1, svbool_t& pg2, svbool_t& pg3) {
	svfloat32_t vout0, vout1, vout2, vout3;
	svfloat32_t va;
	svfloat32_t vb0, vb1, vb2, vb3;
	int out_idx = m*N + n;
	// load output to SVE register
	if (N > 0)
		svfloat32_t vout0 = svdup_n_f32(0.0);
	if (N > 1)
		svfloat32_t vout1 = svdup_n_f32(0.0);
	if (N > 2)
		svfloat32_t vout2 = svdup_n_f32(0.0);
	if (N > 3)
		svfloat32_t vout3 = svdup_n_f32(0.0);

	for (int id_on_cols = start_on_cols; id_on_cols < end_on_cols; id_on_cols++) {
		int k = col[id_on_cols];
		int b_idx = k*N + n;
		// load elem on sparse matrix
		va = svdup_n_f32(value[id_on_cols]);
		// load elems on dense matrix based on the value of N
		if (N > 0)
			vb0 = svld1(pg0, &(mat[b_idx]));
		if (N > 1)
			vb1 = svld1(pg1, &(mat[b_idx + VLEN]));
		if (N > 2)
			vb2 = svld1(pg2, &(mat[b_idx + 2 * VLEN]));
		if (N > 0)
			vb3 = svld1(pg3, &(mat[b_idx + 3 * VLEN]));

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
		svst1(pg1, &(out[out_idx + VLEN]), vout1);
	if (N > 2)
		svst1(pg2, &(out[out_idx + 2 * VLEN]), vout2);
	if (N > 3)
		svst1(pg3, &(out[out_idx + 3 * VLEN]), vout3);
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
