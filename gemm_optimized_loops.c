#include "common.h"

// 函数名: gemm_opt_ikj
// 功能: 实现 ikj 循环顺序的矩阵乘法 C += A * B
//       (调用前需确保 C 矩阵已清零)
// 优化点:
//   - A(i,p) 在内层 j 循环中是不变的，可以预先加载。
//   - B(p,j) 在内层 j 循环中是按行连续访问的，空间局部性好。
void gemm_opt_ikj(int m, int n, int k, double* A, double* B, double* C) {
    // 外层循环: 遍历结果矩阵 C 的每一行
    for (int i = 0; i < m; ++i) {
        // 中层循环: 遍历公共维度 k (对应 A 的列，B 的行)
        for (int p = 0; p < k; ++p) {
            // 预先加载 A(i,p) 的值，因为它在接下来的内层 j 循环中不会改变。
            // 现代编译器通常能很好地将这个值优化到寄存器中。
            double val_A_ip = A_idx(A, i, p, k); // 变量名用英文

            // 内层循环: 遍历结果矩阵 C 的每一列
            for (int j = 0; j < n; ++j) {
                // C(i,j) = C(i,j) + A(i,p) * B(p,j)
                // 在这个循环中，B_idx(B, p, j, n) 访问的是 B 矩阵的第 p 行的元素，
                // 随着 j 的增加，访问是连续的，这非常有利于缓存。
                C_idx(C, i, j, n) += val_A_ip * B_idx(B, p, j, n);
            }
        }
    }
}