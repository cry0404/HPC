#include "common.h" // 引入通用头文件

/**
 * @brief gemm_blocking 函数
 *
 * 功能: 实现分块 (Tiling/Blocking) 矩阵乘法 C = A * B。
 *      分块是一种重要的优化手段，通过将大矩阵划分为小子块进行计算，
 *      可以显著提高缓存的命中率，从而提升性能。
 *      此实现采用 I-K-J 的外层块循环顺序，块内部采用 i-p-j 的元素循环顺序。
 *
 * 参数:
 *   m: 结果矩阵 C 以及输入矩阵 A 的行数。
 *   n: 结果矩阵 C 以及输入矩阵 B 的列数。
 *   k: 输入矩阵 A 的列数，同时也是输入矩阵 B 的行数 (公共维度)。
 *   A: 指向输入矩阵 A 的数据指针 (大小为 m*k)。
 *   B: 指向输入矩阵 B 的数据指针 (大小为 k*n)。
 *   C: 指向结果矩阵 C 的数据指针 (大小为 m*n)。函数将结果累加到 C 中，
 *      调用前 C 应被清零。
 *   block_size: 分块的大小。例如，如果 block_size 为 16，则矩阵会被划分为 16x16 的子块。
 *
 * 注意:
 *   - 边界处理: 当矩阵维度不能被 block_size 整除时，需要正确处理剩余的边缘块。
 *   - 块内乘法: 此处的块内乘法直接展开为三层循环。更高级的实现中，
 *     块内乘法本身也可以是高度优化的微内核(micro-kernel)。
 */
void gemm_blocking(int m, int n, int k, const double* A, const double* B, double* C, int block_size) {
    // 外层循环遍历块的起始索引
    // ii, jj, kk 分别是 C, A, B 矩阵在大矩阵中的块级行或列起始索引

    // 遍历 A 的行块 (ii)，对应 C 的行块
    for (int ii = 0; ii < m; ii += block_size) {
        // 遍历 A 的列块 / B 的行块 (kk)，这是公共维度的分块
        for (int kk = 0; kk < k; kk += block_size) {
            // 遍历 B 的列块 (jj)，对应 C 的列块
            for (int jj = 0; jj < n; jj += block_size) {
                
                // --- 开始处理一个 C_sub(current_m_block, current_n_block)块的计算 ---
                // C_sub += A_sub(current_m_block, current_k_block) * B_sub(current_k_block, current_n_block)
                
                // 计算当前块的实际大小，以处理无法被 block_size 整除的边界情况
                // current_m_block: 当前 C 块和 A 块的实际行数
                int current_m_block = (m - ii < block_size) ? (m - ii) : block_size;
                // current_n_block: 当前 C 块和 B 块的实际列数
                int current_n_block = (n - jj < block_size) ? (n - jj) : block_size;
                // current_k_block: 当前 A 块的实际列数 / B 块的实际行数 (块内的公共维度)
                int current_k_block = (k - kk < block_size) ? (k - kk) : block_size;

                // 内层循环: 对选定的 A, B, C 子块执行矩阵乘法
                // i_block, j_block, p_block 是在当前子块内部的相对索引
                
                // 遍历当前 A 子块的每一行 (i_block)
                for (int i_block = 0; i_block < current_m_block; ++i_block) {
                    // 遍历当前 A 子块的每一列 / B 子块的每一行 (p_block)
                    for (int p_block = 0; p_block < current_k_block; ++p_block) {
                        // 从 A 子块中取出一个元素 A_sub(i_block, p_block)
                        // 它在原始 A 矩阵中的位置是 A(ii + i_block, kk + p_block)
                        // 这个值在接下来的 j_block 循环中是不变的，可以预加载 (尽管简单循环编译器也可能优化)
                        double val_A_ip_block = A_idx(A, ii + i_block, kk + p_block, k);
                        
                        // 遍历当前 B 子块的每一列 (j_block)
                        for (int j_block = 0; j_block < current_n_block; ++j_block) {
                            // 从 B 子块中取出一个元素 B_sub(p_block, j_block)
                            // 它在原始 B 矩阵中的位置是 B(kk + p_block, jj + j_block)
                            double val_B_pj_block = B_idx(B, kk + p_block, jj + j_block, n);
                            
                            // 将乘积累加到 C 子块的对应元素 C_sub(i_block, j_block)
                            // 它在原始 C 矩阵中的位置是 C(ii + i_block, jj + j_block)
                            C_idx(C, ii + i_block, jj + j_block, n) += val_A_ip_block * val_B_pj_block;
                        }
                    }
                }
                // --- 结束处理一个 C_sub 块的计算 ---
            }
        }
    }
} 