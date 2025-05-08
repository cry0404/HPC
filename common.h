#define _POSIX_C_SOURCE 200809L // 为了 clock_gettime 和 CLOCK_MONOTONIC

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>    // 用于标准输入输出，如 printf
#include <stdlib.h>   // 用于内存分配(malloc, free)和随机数(rand, srand)
#include <time.h>     // 用于计时(clock_gettime, time)
#include <math.h>     // 用于数学函数，如 fabs (计算绝对值)
#include <string.h>   // 用于内存操作，如 memset, memcpy

// 矩阵元素访问宏 (行主序存储)
// A 是一个 M x K_dim 的矩阵 (即 K_dim 是 A 的列数)
// B 是一个 K x N_dim 的矩阵 (即 N_dim 是 B 的列数)
// C 是一个 M x N_dim 的矩阵 (即 N_dim 是 C 的列数)
// ptr: 指向矩阵数据开头的指针
// r: 行索引 (0-based)
// c: 列索引 (0-based)
// num_cols: 对应矩阵的总列数，用于计算一维数组中的偏移量

// 这些宏与您现有代码中的 A_idx, B_idx, C_idx 兼容
// A(i,p) 其中 A 是 M x k_glob, k_glob 是 A 的列数
#define A_idx(ptr, r, c, k_glob) ((ptr)[(r) * (k_glob) + (c)])
// B(p,j) 其中 B 是 k_glob x n_glob, n_glob 是 B 的列数
#define B_idx(ptr, r, c, n_glob) ((ptr)[(r) * (n_glob) + (c)])
// C(i,j) 其中 C 是 m_glob x n_glob, n_glob 是 C 的列数
#define C_idx(ptr, r, c, n_glob) ((ptr)[(r) * (n_glob) + (c)])


// --- GEMM 函数声明 ---
// m, n, k 分别是矩阵 C(m,n), A(m,k), B(k,n) 的维度
// A, B 是输入矩阵 (const 表示函数内部不应修改它们)
// C 是输出矩阵 C = C + A * B (通常调用前C需清零)

/**
 * @brief 朴素的 ijk 顺序矩阵乘法。
 * C = A * B (假设 C 初始为0)
 */
void gemm_naive_ijk(int m, int n, int k, const double* A, const double* B, double* C);

/**
 * @brief ikj 循环顺序优化的矩阵乘法。 (来自 gemm_optimized_loops.c)
 * C = A * B (假设 C 初始为0)
 */
void gemm_opt_ikj(int m, int n, int k, const double* A, const double* B, double* C);

/**
 * @brief 分块矩阵乘法。
 * C = A * B (假设 C 初始为0)
 * @param block_size 分块的大小。
 */
void gemm_blocking(int m, int n, int k, const double* A, const double* B, double* C, int block_size);

/**
 * @brief 使用 4x4 微内核的 AVX2 优化封装函数 (当前为标量模拟)。 (来自 gemm_microkernel.c)
 * C = A * B (假设 C 初始为0)
 */
void gemm_microkernel_avx2_wrapper(int m, int n, int k, const double* A, const double* B, double* C);

#endif // COMMON_H