#include "common.h" // 引入通用头文件，它已包含了其他必要的标准库

#define NUM_RUNS 3                 // 每个算法运行的次数，用于取平均值
#define DEFAULT_BLOCK_SIZE 16      // 用于分块算法的默认块大小

// --- 矩阵工具函数 (定义已移至 matrix_utils.c) ---
/*
double* allocate_matrix(int rows, int cols) { ... }
void free_matrix(double* mat) { ... }
void fill_random_matrix(double* mat, int rows, int cols) { ... }
void zero_matrix(double* mat, int rows, int cols) { ... }
double verify_result(const double* C_ref, const double* C_test, int m, int n, int* diff_r, int* diff_c) { ... }
*/

// --- 计时与执行框架 ---

// 定义一个通用的 GEMM 函数指针类型
// 参数: m, n, k, A_ptr, B_ptr, C_ptr
typedef void (*gemm_func_ptr_t)(int, int, int, const double*, const double*, double*);

// 定义一个针对分块优化的 GEMM 函数指针类型
// 参数: m, n, k, A_ptr, B_ptr, C_ptr, block_size
typedef void (*gemm_blocking_func_ptr_t)(int, int, int, const double*, const double*, double*, int);

/**
 * @brief 执行并计时指定的 GEMM 函数。
 * @param name 算法的名称 (用于打印)。
 * @param func 指向标准 GEMM 函数的指针 (如果适用)。
 * @param blocking_func 指向分块 GEMM 函数的指针 (如果适用)。
 * @param m, n, k 矩阵维度。
 * @param A, B 输入矩阵。
 * @param C_test 用于存储当前算法计算结果的矩阵。
 * @param C_ref  用于验证结果的参考矩阵 (通常由朴素算法计算得到)。如果为 NULL，则不进行验证。
 * @param use_blocking_func 布尔标志，如果为 true，则调用 blocking_func，否则调用 func。
 */
void run_and_time_gemm(
    const char* name,
    gemm_func_ptr_t func, 
    gemm_blocking_func_ptr_t blocking_func,
    int m, int n, int k,
    const double* A, const double* B, double* C_test,
    const double* C_ref,
    int use_blocking_func
) {
    double total_time_sec = 0.0;
    printf("--- 运行 %s (维度 M=%d, N=%d, K=%d) ---\n", name, m, n, k);

    for (int run = 0; run < NUM_RUNS; ++run) {
        zero_matrix(C_test, m, n); // 确保每次运行前 C_test 被清零

        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time); // 获取高精度起始时间

        if (use_blocking_func) {
            if (blocking_func) {
                blocking_func(m, n, k, A, B, C_test, DEFAULT_BLOCK_SIZE);
            } else {
                fprintf(stderr, "错误: [%s] 请求调用分块函数，但指针为空。\n", name);
                return;
            }
        } else {
            if (func) {
                func(m, n, k, A, B, C_test);
            } else {
                fprintf(stderr, "错误: [%s] 请求调用标准函数，但指针为空。\n", name);
                return;
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end_time); // 获取高精度结束时间

        double time_taken_sec = (end_time.tv_sec - start_time.tv_sec) +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1e9; // 转换为秒
        total_time_sec += time_taken_sec;
        printf("   迭代 %d 耗时: %.6f 秒\n", run + 1, time_taken_sec);
    }

    double avg_time_sec = total_time_sec / NUM_RUNS;
    double gflops = 0.0;
    if (avg_time_sec > 1e-9 && m > 0 && n > 0 && k > 0) { 
        gflops = (2.0 * (double)m * (double)n * (double)k) / avg_time_sec / 1e9;
    }
    printf("   平均耗时: %.6f 秒, GFLOPS: %.2f\n", avg_time_sec, gflops);

    if (C_ref) {
        int diff_r, diff_c;
        double max_diff = verify_result(C_ref, C_test, m, n, &diff_r, &diff_c);
        if (diff_r != -1 && max_diff > 1e-9) { 
            printf("   验证: 结果最大差异 = %e 在位置 C(%d,%d)\n", max_diff, diff_r, diff_c);
        } else if (max_diff <= 1e-9) {
             printf("   验证: 结果与参考值一致 (最大差异 <= 1e-9)。\n");
        } else { 
             printf("   验证: 结果与参考值完全一致。\n");
        }
    }
    printf("\n"); 
}


// --- 主函数 ---
int main() {
    srand(time(NULL));

    printf("矩阵乘法性能测试 (交互式输入维度)\n");
    printf("=====================================================\n");
    printf("每个算法将针对每个维度运行 %d 次并计算平均性能。\n", NUM_RUNS);
    printf("分块优化使用的块大小为: %d\n", DEFAULT_BLOCK_SIZE);
    printf("微内核优化当前使用的是其 4x4 真实SIMD 版本。\n");
    printf("=====================================================\n\n");

    char more_tests = 'y';
    while (more_tests == 'y' || more_tests == 'Y') {
        int m, n, k;
        printf("请输入矩阵维度 M N K (以空格分隔, 例如: 100 100 100): ");
        if (scanf("%d %d %d", &m, &n, &k) != 3) {
            printf("输入错误，请输入三个整数值。\n");
            while (getchar() != '\n'); 
            continue; 
        }

        if (m <= 0 || n <= 0 || k <= 0) {
            printf("错误：矩阵维度 M, N, K 必须是正整数。\n");
            continue;
        }

        printf("\n开始测试维度: M=%d, N=%d, K=%d\n", m, n, k);
        printf("-----------------------------------------------------\n");

        double* A      = allocate_matrix(m, k);
        double* B      = allocate_matrix(k, n);
        double* C_ref  = allocate_matrix(m, n); 
        double* C_test = allocate_matrix(m, n); 

        fill_random_matrix(A, m, k);
        fill_random_matrix(B, k, n);

        run_and_time_gemm("0. 朴素 ijk 实现", 
                          gemm_naive_ijk, NULL, 
                          m, n, k, A, B, C_ref, 
                          NULL, 0); 

        run_and_time_gemm("1. 循环顺序优化 ikj",
                          gemm_opt_ikj, NULL,
                          m, n, k, A, B, C_test,
                          C_ref, 0); 

        run_and_time_gemm("2. 分块优化",
                          NULL, gemm_blocking, 
                          m, n, k, A, B, C_test,
                          C_ref, 1); 

        run_and_time_gemm("3. 微内核 AVX2 (4x4 真实SIMD)", 
                          gemm_microkernel_avx2_wrapper, NULL,
                          m, n, k, A, B, C_test,
                          C_ref, 0);

        run_and_time_gemm("4. Packing + AVX2 微内核",
                          gemm_packed_avx2, NULL,
                          m, n, k, A, B, C_test,
                          C_ref, 0);
        
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_ref);
        free_matrix(C_test);

        printf("维度 M=%d, N=%d, K=%d 测试完成。\n", m, n, k);
        printf("=====================================================\n");

        printf("是否要测试其他维度? (y/n): ");
        while (getchar() != '\n'); 
        if (scanf(" %c", &more_tests) != 1) {
            fprintf(stderr, "输入错误，将退出测试。\n");
            more_tests = 'n';
        }
        while (getchar() != '\n'); 
        printf("\n");
    }

    printf("所有测试已完成。\n");
    return 0;
} 