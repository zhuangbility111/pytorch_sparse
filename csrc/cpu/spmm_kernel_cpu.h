#pragma once

#ifdef __ARM_FEATURE_SVE
	#include <arm_sve.h>
	#define VLEN 16
#endif

// typedef void (*inner_kernel)(int, float *, float *, float *, int);
using Inner_kernel = void(*)(float*, float*, float*, float*,
							 int, int, int, int, int,
							 svbool_t&, svbool_t&, svbool_t&, svbool_t&);

Inner_kernel get_kernel_1xN(int n);
