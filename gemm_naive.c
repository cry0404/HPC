#include "common.h" // 引入通用头文件，包含 A_idx, B_idx, C_idx 宏和基本库

/**
 * @brief gemm_naive_ijk 函数
 *
 * 功能: 实现最朴素的 ijk 循环顺序的矩阵乘法 C = A * B。
 *      这个版本通常作为性能基准，因为它的数据局部性较差，
 *      尤其是在 B 矩阵的访问上（按列访问，如果行主序存储则不连续）。
 *
 * 参数:
 *   m: 结果矩阵 C 以及输入矩阵 A 的行数。
 *   n: 结果矩阵 C 以及输入矩阵 B 的列数。
 *   k: 输入矩阵 A 的列数，同时也是输入矩阵 B 的行数 (公共维度)。
 *   A: 指向输入矩阵 A 的数据指针 (大小为 m*k)。矩阵 A 被视为 const，函数内部不应修改它。
 *   B: 指向输入矩阵 B 的数据指针 (大小为 k*n)。矩阵 B 被视为 const。
 *   C: 指向结果矩阵 C 的数据指针 (大小为 m*n)。此函数计算 A*B 并将结果累加到 C 中，
 *      因此，调用此函数前，矩阵 C 应该已经被正确初始化 (通常是清零)。
 *
 * 注意: 宏 A_idx, B_idx, C_idx 用于将二维的矩阵索引 (row, col) 转换为一维数组的索引。
 *       例如 A_idx(A, i, p, k) 访问 A 矩阵的第 i 行第 p 列，其中 k 是 A 矩阵的总列数。
 */
void gemm_naive_ijk(int m, int n, int k, const double* A, const double* B, double* C) {
    // 外层循环: 遍历结果矩阵 C 的每一行 (i from 0 to m-1)
    for (int i = 0; i < m; ++i) {
        // 中层循环: 遍历结果矩阵 C 的每一列 (j from 0 to n-1)
        for (int j = 0; j < n; ++j) {
            // 此处 C(i,j) 的计算依赖于 A 的第 i 行和 B 的第 j 列的点积
            // double sum = 0; // 如果C没有预先清零，可以在这里初始化
            
            // 内层循环: 遍历公共维度 k (p from 0 to k-1)
            // 这个循环计算 C(i,j) = sum_{p=0}^{k-1} A(i,p) * B(p,j)
            for (int p = 0; p < k; ++p) {
                // C(i,j) += A(i,p) * B(p,j)
                C_idx(C, i, j, n) += A_idx(A, i, p, k) * B_idx(B, p, j, n);
            }
            // C_idx(C, i, j, n) = sum; // 如果在这里初始化sum
        }
    }
}