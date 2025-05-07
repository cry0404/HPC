#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>  // 用于 printf
#include <stdlib.h> // 用于 malloc, free, rand, srand, atoi
#include <time.h>   // 用于 clock (简易计时)
#include <math.h>   // 用于 fabs (浮点数绝对值)
#include <assert.h> // 用于 assert (调试断言)
#include <string.h> // 用于 memset (内存清零)

#ifdef __linux__
#include <sys/time.h> // 包含 gettimeofday
double get_time();
#else // 其他系统使用标准 clock()，精度较低
double get_time();
#endif

// 函数声明
void init_matrix(int rows, int cols, double* matrix);
void verify_gemm(int m_rows, int n_cols, double* C_computed, double* C_reference);

// 宏：方便地访问行主序存储的矩阵元素
// k_colsA 是 A 矩阵的列数 (即 k)
// n_colsB 是 B 矩阵的列数 (即 n)
// n_colsC 是 C 矩阵的列数 (即 n)
#define A_idx(matrix_ptr, r, c, k_colsA) (matrix_ptr)[(r)*(k_colsA) + (c)]
#define B_idx(matrix_ptr, r, c, n_colsB) (matrix_ptr)[(r)*(n_colsB) + (c)]
#define C_idx(matrix_ptr, r, c, n_colsC) (matrix_ptr)[(r)*(n_colsC) + (c)]

// 宏：取两个数的较小值 (用于分块的边界处理)
#define MIN(a,b) (((a)<(b))?(a):(b))

#endif // COMMON_H