#include "common.h"

// 函数名: gemm_blocked
// 功能: 实现分块矩阵乘法 C += A * B
//       (调用前需确保 C 矩阵已清零)
// 参数:
//   block_size: 用于分块的正方形块的边长
// 优化点:
//   - 将计算分解为针对小数据块的操作，提高缓存命中率。
//   - 内部块的计算可以使用前面已优化的循环顺序 (如 ikj)。
void gemm_blocked(int m, int n, int k, double* A, double* B, double* C, int block_size) { // 参数名用英文
    // 外三层循环，按 block_size 步长遍历整个矩阵，定义当前处理的“大块”
    for (int i0 = 0; i0 < m; i0 += block_size) {         // C 和 A 的行块起始索引
        for (int j0 = 0; j0 < n; j0 += block_size) {     // C 和 B 的列块起始索引
            // 对于当前的 C 子块 C_sub(i0,j0)，它需要累加所有对应的 A_sub(i0,p0) * B_sub(p0,j0)
            // 由于 C 在 main.c 中已经全局清零，这里直接进行累加操作。
            for (int p0 = 0; p0 < k; p0 += block_size) { // A 的列块 / B 的行块起始索引

                // 内三层循环，计算当前三个小块之间的乘积累加
                // C_sub(i,j) += A_sub(i,p) * B_sub(p,j)
                // MIN 宏用于处理矩阵边缘，防止索引越界。
                // 块内计算采用 ikj 类似的思路：
                for (int i = i0; i < MIN(i0 + block_size, m); ++i) { // 遍历当前 C_sub 和 A_sub 的行
                    for (int p = p0; p < MIN(p0 + block_size, k); ++p) { // 遍历当前 A_sub 的列 / B_sub 的行
                        double val_A_ip = A_idx(A, i, p, k); // 预加载 A(i,p)

                        for (int j = j0; j < MIN(j0 + block_size, n); ++j) { // 遍历当前 C_sub 和 B_sub 的列
                            C_idx(C, i, j, n) += val_A_ip * B_idx(B, p, j, n);
                        }
                    }
                }
            }
        }
    }
}