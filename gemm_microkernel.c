#include "common.h"     // 引入通用宏和函数
#include <immintrin.h>  // 引入 AVX/AVX2 等 SIMD 内建函数的头文件

// 定义微内核处理的 C 矩阵小块的维度 (MR_micro x NR_micro)
// 对于 AVX2 处理 double (64位)，一个 __m256d 寄存器可以存 4 个 double。
// 我们设计一个 4x4 的微内核。
// #define MR_AVX2 4      // 已移至 common.h
// #define NR_AVX2 4      // 已移至 common.h

// AVX2 微内核，计算 4x4 的 C += A * B
// C (4x4) <double>
// A (4xk_common) <double>
// B (k_commonx4) <double>
inline void gemm_kernel_4x4_avx2(
    int k_common,                // 公共维度 k
    const double* ptr_A_row_start, // 指向 A 子块的起始行 (即 A[i_micro][0])
    int lda,                     // 矩阵 A 的主维度 (即原始矩阵 A 的列数 k)
    const double* ptr_B_col_start, // 指向 B 子块的起始列 (即 B[0][j_micro])
    int ldb,                     // 矩阵 B 的主维度 (即原始矩阵 B 的列数 n)
    double* ptr_C_target,        // 指向 C 子块的起始位置 (即 C[i_micro][j_micro])
    int ldc                      // 矩阵 C 的主维度 (即原始矩阵 C 的列数 n)
) {
    // AVX2 寄存器用于存储 C 矩阵的 4 行 (每行4个double)
    __m256d c_row0_vec, c_row1_vec, c_row2_vec, c_row3_vec;

    // 1. 加载 C 矩阵的 4x4 子块到 AVX 寄存器
    c_row0_vec = _mm256_loadu_pd(ptr_C_target + 0 * ldc); 
    c_row1_vec = _mm256_loadu_pd(ptr_C_target + 1 * ldc); 
    c_row2_vec = _mm256_loadu_pd(ptr_C_target + 2 * ldc); 
    c_row3_vec = _mm256_loadu_pd(ptr_C_target + 3 * ldc); 

    // 2. 主循环，遍历公共维度 k_common
    for (int p = 0; p < k_common; ++p) {
        // 加载 B 矩阵的一行 (4个元素) B[p][j_micro ... j_micro+3]
        __m256d b_p_j_vec = _mm256_loadu_pd(ptr_B_col_start + p * ldb);

        // 加载 A 矩阵的一列中的4个元素，并广播 (broadcast)
        __m256d a_i0_p_vec = _mm256_broadcast_sd(ptr_A_row_start + 0 * lda + p); 
        __m256d a_i1_p_vec = _mm256_broadcast_sd(ptr_A_row_start + 1 * lda + p); 
        __m256d a_i2_p_vec = _mm256_broadcast_sd(ptr_A_row_start + 2 * lda + p); 
        __m256d a_i3_p_vec = _mm256_broadcast_sd(ptr_A_row_start + 3 * lda + p); 

        // 执行 FMA (Fused Multiply-Add) 操作: C_row += A_val * B_row_vec
        c_row0_vec = _mm256_fmadd_pd(a_i0_p_vec, b_p_j_vec, c_row0_vec);
        c_row1_vec = _mm256_fmadd_pd(a_i1_p_vec, b_p_j_vec, c_row1_vec);
        c_row2_vec = _mm256_fmadd_pd(a_i2_p_vec, b_p_j_vec, c_row2_vec);
        c_row3_vec = _mm256_fmadd_pd(a_i3_p_vec, b_p_j_vec, c_row3_vec);
    }

    // 3. 将更新后的 C 子块从 AVX 寄存器存回内存
    _mm256_storeu_pd(ptr_C_target + 0 * ldc, c_row0_vec);
    _mm256_storeu_pd(ptr_C_target + 1 * ldc, c_row1_vec);
    _mm256_storeu_pd(ptr_C_target + 2 * ldc, c_row2_vec);
    _mm256_storeu_pd(ptr_C_target + 3 * ldc, c_row3_vec);
}

// 内联函数：不使用AVX2，使用纯标量计算实现4x4内核
inline void gemm_kernel_4x4_scalar(
    int k_inner,
    const double* ptr_A_panel, int k_cols_A,
    const double* ptr_B_panel, int n_cols_B,
    double* ptr_C_target, int n_cols_C
) {
    // 预先加载C矩阵的值
    double c00 = C_idx(ptr_C_target, 0, 0, n_cols_C);
    double c01 = C_idx(ptr_C_target, 0, 1, n_cols_C);
    double c02 = C_idx(ptr_C_target, 0, 2, n_cols_C);
    double c03 = C_idx(ptr_C_target, 0, 3, n_cols_C);
    
    double c10 = C_idx(ptr_C_target, 1, 0, n_cols_C);
    double c11 = C_idx(ptr_C_target, 1, 1, n_cols_C);
    double c12 = C_idx(ptr_C_target, 1, 2, n_cols_C);
    double c13 = C_idx(ptr_C_target, 1, 3, n_cols_C);
    
    double c20 = C_idx(ptr_C_target, 2, 0, n_cols_C);
    double c21 = C_idx(ptr_C_target, 2, 1, n_cols_C);
    double c22 = C_idx(ptr_C_target, 2, 2, n_cols_C);
    double c23 = C_idx(ptr_C_target, 2, 3, n_cols_C);
    
    double c30 = C_idx(ptr_C_target, 3, 0, n_cols_C);
    double c31 = C_idx(ptr_C_target, 3, 1, n_cols_C);
    double c32 = C_idx(ptr_C_target, 3, 2, n_cols_C);
    double c33 = C_idx(ptr_C_target, 3, 3, n_cols_C);
    
    // 内层循环遍历k维度
    for (int p = 0; p < k_inner; ++p) {
        // 加载A的4个元素
        double a0 = A_idx(ptr_A_panel, 0, p, k_cols_A);
        double a1 = A_idx(ptr_A_panel, 1, p, k_cols_A);
        double a2 = A_idx(ptr_A_panel, 2, p, k_cols_A);
        double a3 = A_idx(ptr_A_panel, 3, p, k_cols_A);
        
        // 加载B的4个元素
        double b0 = B_idx(ptr_B_panel, p, 0, n_cols_B);
        double b1 = B_idx(ptr_B_panel, p, 1, n_cols_B);
        double b2 = B_idx(ptr_B_panel, p, 2, n_cols_B);
        double b3 = B_idx(ptr_B_panel, p, 3, n_cols_B);
        
        // 计算累加
        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;
        
        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;
        
        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;
        
        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
    }
    
    // 存回C矩阵
    C_idx(ptr_C_target, 0, 0, n_cols_C) = c00;
    C_idx(ptr_C_target, 0, 1, n_cols_C) = c01;
    C_idx(ptr_C_target, 0, 2, n_cols_C) = c02;
    C_idx(ptr_C_target, 0, 3, n_cols_C) = c03;
    
    C_idx(ptr_C_target, 1, 0, n_cols_C) = c10;
    C_idx(ptr_C_target, 1, 1, n_cols_C) = c11;
    C_idx(ptr_C_target, 1, 2, n_cols_C) = c12;
    C_idx(ptr_C_target, 1, 3, n_cols_C) = c13;
    
    C_idx(ptr_C_target, 2, 0, n_cols_C) = c20;
    C_idx(ptr_C_target, 2, 1, n_cols_C) = c21;
    C_idx(ptr_C_target, 2, 2, n_cols_C) = c22;
    C_idx(ptr_C_target, 2, 3, n_cols_C) = c23;
    
    C_idx(ptr_C_target, 3, 0, n_cols_C) = c30;
    C_idx(ptr_C_target, 3, 1, n_cols_C) = c31;
    C_idx(ptr_C_target, 3, 2, n_cols_C) = c32;
    C_idx(ptr_C_target, 3, 3, n_cols_C) = c33;
}

// 函数：使用朴素 ikj 方法处理边界情况
static void gemm_scalar_boundary(
    int m_start, int m_end, 
    int n_start, int n_end,
    int k, const double* A, const double* B, double* C, int n_cols_C_total // Renamed 'n' to avoid conflict if k is also named n
) {
    for (int i = m_start; i < m_end; ++i) {
        for (int p_loop = 0; p_loop < k; ++p_loop) { // Renamed 'p' to 'p_loop'
            double val_A_ip = A_idx(A, i, p_loop, k);
            for (int j = n_start; j < n_end; ++j) {
                C_idx(C, i, j, n_cols_C_total) += val_A_ip * B_idx(B, p_loop, j, n_cols_C_total);
            }
        }
    }
}

// 函数名: gemm_microkernel_avx2_wrapper
// 功能: 一个简单的包装函数，用于演示如何调用 4x4 微内核。
//       它按 MR_AVX2 x NR_AVX2 的步长遍历 C 矩阵，并为每个微块调用内核。
//       C += A * B (调用前需确保 C 矩阵已清零)
// 参数:
//   m, n, k_orig: 原始大矩阵的维度
//   A, B, C: 指向原始大矩阵的指针
void gemm_microkernel_avx2_wrapper(int m, int n_orig, int k_orig, const double* A, const double* B, double* C) {
    // 外层循环按微内核的块大小步进，遍历 C 矩阵
    // 处理内部完整块（可以被 MR_AVX2 和 NR_AVX2 整除的部分）
    int m_boundary = (m / MR_AVX2) * MR_AVX2;
    int n_boundary = (n_orig / NR_AVX2) * NR_AVX2;
    
    // 处理完整块
    for (int i_micro = 0; i_micro < m_boundary; i_micro += MR_AVX2) {
        for (int j_micro = 0; j_micro < n_boundary; j_micro += NR_AVX2) {
            // 调用真正的 4x4 AVX2 内核
            gemm_kernel_4x4_avx2(
                k_orig,                             // k_common: 公共维度 k_orig
                &A_idx(A, i_micro, 0, k_orig),      // ptr_A_row_start: 指向 A(i_micro, 0)
                k_orig,                             // lda: A 的主维度 (列数 k_orig)
                &B_idx(B, 0, j_micro, n_orig),      // ptr_B_col_start: 指向 B(0, j_micro)
                n_orig,                             // ldb: B 的主维度 (列数 n_orig)
                &C_idx(C, i_micro, j_micro, n_orig),// ptr_C_target: 指向 C(i_micro, j_micro)
                n_orig                              // ldc: C 的主维度 (列数 n_orig)
            );
        }
    }
    
    // 处理剩余边界（M方向）- 仍然使用标量边界处理
    if (m_boundary < m) {
        for (int j_col_block_start = 0; j_col_block_start < n_boundary; j_col_block_start += NR_AVX2) {
            gemm_scalar_boundary(m_boundary, m, 
                                 j_col_block_start, j_col_block_start + NR_AVX2, 
                                 k_orig, A, B, C, n_orig);
        }
    }
    
    // 处理剩余边界（N方向）- 仍然使用标量边界处理
    if (n_boundary < n_orig) {
        for (int i_row_block_start = 0; i_row_block_start < m_boundary; i_row_block_start += MR_AVX2) {
            gemm_scalar_boundary(i_row_block_start, i_row_block_start + MR_AVX2, 
                                 n_boundary, n_orig, 
                                 k_orig, A, B, C, n_orig);
        }
    }
    
    // 处理角落部分（既是M边界又是N边界）- 仍然使用标量边界处理
    if (m_boundary < m && n_boundary < n_orig) {
        gemm_scalar_boundary(m_boundary, m, n_boundary, n_orig, k_orig, A, B, C, n_orig);
    }
}