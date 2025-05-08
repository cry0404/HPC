#include "common.h"
#include <immintrin.h> // 即使 gemm_kernel_4x4_avx2 已在 common.h 中通过 gemm_microkernel.c 声明，此处包含仍是好习惯
#include <stdio.h>   // 如果以后添加调试或详细模式的 printf，会用到

// --- 用于 Packing 和宏内核的块大小定义 ---
// MC, NC: C矩阵的M和N维度的外层分块大小。
// KC: K维度 (公共维度) 的外层分块大小。
// 理想情况下，它们应该是 MR_AVX2 和 NR_AVX2 的倍数，或者根据缓存大小进行调优。
#define GEMM_PACKED_MC 64
#define GEMM_PACKED_NC 64 
#define GEMM_PACKED_KC 256

// 微内核维度 (已在 gemm_microkernel.c 中定义，但此处为清晰起见提及)
// #define MR_AVX2 4
// #define NR_AVX2 4
// 我们将依赖 common.h 中的 MR_AVX2 和 NR_AVX2 (如果决定将定义移到那里) 或直接使用 4。
// 目前，如果 common.h 作用域内找不到，则在此文件内部硬编码 MR/NR 为 4。
// 实际上，gemm_kernel_4x4_avx2 特定于4x4，因此 MR=4, NR=4 由其固定。

/**
 * @brief 将矩阵 A 的一个面板 (panel) 打包 (pack) 到一个连续的缓冲区中 (行主序)。
 *
 * @param packed_A 指向目标打包缓冲区的指针。
 * @param A 指向源矩阵 A 的指针。
 * @param num_rows 要从 A 打包的行数 (当前的 mc 大小)。
 * @param num_cols 要从 A 打包的列数 (当前的 kc 大小)。
 * @param row_offset 在原始矩阵 A 中的起始行。
 * @param col_offset 在原始矩阵 A 中的起始列。
 * @param lda_original 原始矩阵 A 的主维度 (列数)。
 */
static void pack_A_panel(double* packed_A, const double* A, 
                         int num_rows, int num_cols, 
                         int row_offset, int col_offset, int lda_original) {
    for (int i = 0; i < num_rows; ++i) {
        // 对于 A 的每一行中的元素，将其连续存放到 packed_A 中
        for (int p = 0; p < num_cols; ++p) {
            // packed_A 是一个 num_rows x num_cols 的行主序矩阵
            packed_A[i * num_cols + p] = A_idx(A, row_offset + i, col_offset + p, lda_original);
        }
    }
}

/**
 * @brief 将矩阵 B 的一个面板打包到连续缓冲区中 (行主序)。
 *        注意：高性能库通常会将B打包成列主序或一种特殊格式，以利于微内核的列访问或SIMD操作。
 *        此处为了简单，仍然按行主序打包。
 *
 * @param packed_B 指向目标打包缓冲区的指针。
 * @param B 指向源矩阵 B 的指针。
 * @param num_rows 要从 B 打包的行数 (当前的 kc 大小)。
 * @param num_cols 要从 B 打包的列数 (当前的 nc 大小)。
 * @param row_offset 在原始矩阵 B 中的起始行。
 * @param col_offset 在原始矩阵 B 中的起始列。
 * @param ldb_original 原始矩阵 B 的主维度 (列数)。
 */
static void pack_B_panel(double* packed_B, const double* B,
                         int num_rows, int num_cols,
                         int row_offset, int col_offset, int ldb_original) {
    for (int p = 0; p < num_rows; ++p) {
        // 对于 B 的每一行中的元素，将其连续存放到 packed_B 中
        for (int j = 0; j < num_cols; ++j) {
            // packed_B 是一个 num_rows x num_cols 的行主序矩阵
            packed_B[p * num_cols + j] = B_idx(B, row_offset + p, col_offset + j, ldb_original);
        }
    }
}

/**
 * @brief 对已经打包好的 A 和 B 面板执行标量的 GEMM 操作 C += A*B。
 *        用于处理打包面板边缘处不能被微内核完整覆盖的任意大小的小块。
 *        A 是 (m_sub x k_sub), B 是 (k_sub x n_sub), C 是 (m_sub x n_sub)
 * 
 * @param m_sub C 和 A 的子块的行数。
 * @param n_sub C 和 B 的子块的列数。
 * @param k_sub A 和 B 的子块的公共维度。
 * @param packed_A_sub_start 指向打包后 A 子块的起始指针。
 * @param lda_packed 打包后 A 子块的主维度 (即 k_sub)。
 * @param packed_B_sub_start 指向打包后 B 子块的起始指针。
 * @param ldb_packed 打包后 B 子块的主维度 (即 n_sub)。
 * @param C_target_start 指向原始 C 矩阵中目标子块的起始指针。
 * @param ldc_original 原始 C 矩阵的主维度。
 */
static void gemm_scalar_partial_on_packed(
    int m_sub, int n_sub, int k_sub,
    const double* packed_A_sub_start, int lda_packed, // lda_packed 对于行主序的 packed_A_sub_start 实际上是其列数 k_sub
    const double* packed_B_sub_start, int ldb_packed, // ldb_packed 对于行主序的 packed_B_sub_start 实际上是其列数 n_sub
    double* C_target_start, int ldc_original) {

    for (int i = 0; i < m_sub; ++i) {
        for (int j = 0; j < n_sub; ++j) {
            double sum = 0.0; // C(i,j) 是 A 的第i行与B的第j列的点积累加器
            for (int p = 0; p < k_sub; ++p) {
                // packed_A_sub_start[i * lda_packed + p] 访问打包A中的元素 A_pack[i][p]
                // packed_B_sub_start[p * ldb_packed + j] 访问打包B中的元素 B_pack[p][j]
                sum += packed_A_sub_start[i * lda_packed + p] * packed_B_sub_start[p * ldb_packed + j];
            }
            // 将计算结果累加到原始C矩阵的对应位置
            C_idx(C_target_start, i, j, ldc_original) += sum;
        }
    }
}

// AVX2 微内核的前向声明 (在 gemm_microkernel.c 中定义, 在 common.h 中声明)
// 如果 common.h 没有提供，我们需要在这里有完整的签名。
// 假设它通过 common.h 可用或直接链接。
// void gemm_kernel_4x4_avx2(...); // 这应该来自 common.h

// 我们需要 MR_AVX2 和 NR_AVX2，假设它们是4，或者通过 common.h 从 gemm_microkernel.c 定义
// 为安全起见，如果不可用，则使用4。
// 这些宏已移至 common.h，所以这里的 #ifndef/#define 仅作后备，通常不应被触发。
#ifndef MR_AVX2
#warning "MR_AVX2 not defined in common.h, defaulting to 4 in gemm_packing.c"
#define MR_AVX2 4
#endif
#ifndef NR_AVX2
#warning "NR_AVX2 not defined in common.h, defaulting to 4 in gemm_packing.c"
#define NR_AVX2 4
#endif


/**
 * @brief 使用 Packing 和 AVX2 微内核执行 GEMM C += A * B。
 * 
 * @param m 矩阵 C 和 A 的总行数。
 * @param n 矩阵 C 和 B 的总列数。
 * @param k 矩阵 A 的总列数 / B 的总行数。
 * @param A 指向输入矩阵 A 的指针。
 * @param B 指向输入矩阵 B 的指针。
 * @param C_orig 指向输出/累加矩阵 C 的指针。
 */
void gemm_packed_avx2(int m, int n, int k, const double* A, const double* B, double* C_orig) {
    // 分配打包缓冲区
    // 为简单起见，根据定义的宏分配可能的最大尺寸。
    // 在真实的库中，这些缓冲区可能会有不同的管理方式，或者是线程局部的。
    double* packed_A_buffer = (double*)malloc(GEMM_PACKED_MC * GEMM_PACKED_KC * sizeof(double));
    double* packed_B_buffer = (double*)malloc(GEMM_PACKED_KC * GEMM_PACKED_NC * sizeof(double));

    if (!packed_A_buffer || !packed_B_buffer) {
        fprintf(stderr, "错误：无法分配打包缓冲区。打包版本将跳过执行。\n");
        if (packed_A_buffer) free(packed_A_buffer);
        if (packed_B_buffer) free(packed_B_buffer);
        // 可以选择调用一个非打包版本作为后备，或者直接返回让GFLOPS显示为0。
        return;
    }

    // 最外层循环：按 GEMM_PACKED_NC, GEMM_PACKED_KC, GEMM_PACKED_MC 的块大小遍历 C
    // BLIS库的典型循环顺序是 jc, pc, ic (对应 n, k, m)
    for (int jc = 0; jc < n; jc += GEMM_PACKED_NC) { // 遍历 C 的列块 (NC)
        int current_nc = (n - jc < GEMM_PACKED_NC) ? (n - jc) : GEMM_PACKED_NC;

        for (int pc = 0; pc < k; pc += GEMM_PACKED_KC) { // 遍历公共维度 K 的块 (KC)
            int current_kc = (k - pc < GEMM_PACKED_KC) ? (k - pc) : GEMM_PACKED_KC;

            // 打包 B 的面板：current_kc 行, current_nc 列
            // 源 B 面板起始于 B[pc, jc]
            // 注意：对于B，我们期望打包后的数据能被微内核按列（或按行向量）高效访问。
            // 当前 pack_B_panel 是按行主序打包 B 的 KCxNC 子块。
            pack_B_panel(packed_B_buffer, B, current_kc, current_nc, pc, jc, n);

            for (int ic = 0; ic < m; ic += GEMM_PACKED_MC) { // 遍历 C 的行块 (MC)
                int current_mc = (m - ic < GEMM_PACKED_MC) ? (m - ic) : GEMM_PACKED_MC;

                // 打包 A 的面板：current_mc 行, current_kc 列
                // 源 A 面板起始于 A[ic, pc]
                pack_A_panel(packed_A_buffer, A, current_mc, current_kc, ic, pc, k);

                // --- 宏内核：处理 packed_A_buffer 和 packed_B_buffer ---
                // 迭代处理 C_orig[ic, jc] 对应的 current_mc x current_nc 子矩阵
                // 使用 packed_A_buffer (current_mc x current_kc) 
                // 和 packed_B_buffer (current_kc x current_nc)。
                for (int i_micro = 0; i_micro < current_mc; i_micro += MR_AVX2) { // 遍历打包A的行 (MR步进)
                    int m_sub_panel = (current_mc - i_micro < MR_AVX2) ? (current_mc - i_micro) : MR_AVX2;

                    for (int j_micro = 0; j_micro < current_nc; j_micro += NR_AVX2) { // 遍历打包B的列 (NR步进)
                        int n_sub_panel = (current_nc - j_micro < NR_AVX2) ? (current_nc - j_micro) : NR_AVX2;
                        
                        // 计算C中当前4x4 (或更小) 微块的起始指针
                        double* C_target_ptr = &C_idx(C_orig, ic + i_micro, jc + j_micro, n);

                        if (m_sub_panel == MR_AVX2 && n_sub_panel == NR_AVX2) {
                            // 如果是完整的 4x4 块，调用 AVX2 微内核
                            // gemm_kernel_4x4_avx2 对 C 进行累加: C += A_pack * B_pack
                            gemm_kernel_4x4_avx2(
                                current_kc,                                 // 公共维度 current_kc (打包块的内维度)
                                packed_A_buffer + i_micro * current_kc,     // 指向打包A中当前微块的起始行: A_pack[i_micro][0]
                                current_kc,                                 // 打包A的主维度 (列数 current_kc)
                                packed_B_buffer + 0 * current_nc + j_micro, // 指向打包B中当前微块的起始列: B_pack[0][j_micro]
                                                                            // (对于行主序的packed_B, B_pack[p_idx * current_nc + j_micro] 是第p_idx行的第j_micro列元素)
                                                                            // ptr_B_col_start 在微内核中是以 B_pack[0][j_micro] 为基准按行读取，这是对的。
                                current_nc,                                 // 打包B的主维度 (列数 current_nc)
                                C_target_ptr,                               // 指向原始C矩阵中子块 C_orig[ic+i_micro, jc+j_micro]
                                n                                           // 原始C矩阵的主维度 (总列数)
                            );
                        } else {
                            // 处理打包面板边缘处不能被4x4整除的子块
                            // 使用标量代码在打包数据上进行计算
                            gemm_scalar_partial_on_packed(
                                m_sub_panel, n_sub_panel, current_kc,
                                packed_A_buffer + i_micro * current_kc,     // 指向打包A中当前子块的起始
                                current_kc,                                 // 打包A的主维度
                                packed_B_buffer + 0 * current_nc + j_micro, // 指向打包B中当前子块的起始
                                current_nc,                                 // 打包B的主维度
                                C_target_ptr,                               // 指向原始C矩阵
                                n                                           // 原始C矩阵的主维度
                            );
                        }
                    }
                } // 结束微内核循环 (i_micro, j_micro)
            } // 结束 ic 循环 (GEMM_PACKED_MC)
        } // 结束 pc 循环 (GEMM_PACKED_KC)
    } // 结束 jc 循环 (GEMM_PACKED_NC)

    free(packed_A_buffer);
    free(packed_B_buffer);
} 