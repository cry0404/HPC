#include "common.h" // 引入通用头文件，它已包含了其他必要的标准库

#define NUM_RUNS 3                 // 每个算法运行的次数，用于取平均值
#define DEFAULT_BLOCK_SIZE 16      // 用于分块算法的默认块大小

// --- 矩阵工具函数 ---

/**
 * @brief 分配矩阵所需的内存。
 * @param rows 矩阵的行数。
 * @param cols 矩阵的列数。
 * @return 指向分配的内存块的指针。如果分配失败则程序退出。
 */
double* allocate_matrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (!mat) {
        perror("错误：无法分配矩阵内存");
        exit(EXIT_FAILURE);
    }
    return mat;
}

/**
 * @brief 释放之前分配的矩阵内存。
 * @param mat 指向要释放的矩阵内存的指针。
 */
void free_matrix(double* mat) {
    if (mat) {
        free(mat);
    }
}

/**
 * @brief 使用 0.0 到 1.0 之间的随机浮点数填充矩阵。
 * @param mat 指向要填充的矩阵。
 * @param rows 矩阵的行数。
 * @param cols 矩阵的列数。
 */
void fill_random_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (double)rand() / RAND_MAX; // 生成 [0.0, 1.0] 范围的随机数
    }
}

/**
 * @brief 将矩阵的所有元素设置为零。
 * @param mat 指向要清零的矩阵。
 * @param rows 矩阵的行数。
 * @param cols 矩阵的列数。
 */
void zero_matrix(double* mat, int rows, int cols) {
    memset(mat, 0, rows * cols * sizeof(double));
}

/**
 * @brief 验证两个矩阵是否足够接近。
 *        计算 C_ref 和 C_test 之间每个对应元素差的绝对值的最大值。
 * @param C_ref 指向参考结果矩阵。
 * @param C_test 指向测试结果矩阵。
 * @param m 矩阵的行数。
 * @param n 矩阵的列数。
 * @param diff_r (输出参数) 如果发现差异，存储最大差异处的行索引。
 * @param diff_c (输出参数) 如果发现差异，存储最大差异处的列索引。
 * @return 返回两个矩阵之间的最大绝对差异值。
 */
double verify_result(const double* C_ref, const double* C_test, int m, int n, int* diff_r, int* diff_c) {
    double max_diff = 0.0;
    *diff_r = -1; // 初始表示未找到差异位置
    *diff_c = -1;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // 使用 C_idx 宏访问元素，确保与 GEMM 函数内部一致
            double ref_val = C_idx(C_ref, i, j, n);
            double test_val = C_idx(C_test, i, j, n);
            double diff = fabs(ref_val - test_val);

            if (diff > max_diff) {
                max_diff = diff;
                *diff_r = i;
                *diff_c = j;
            }
        }
    }
    return max_diff;
}

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
    // GFLOPS = (2 * M * N * K) / (平均时间 * 10^9)
    // 注意: 对于非常小的矩阵，M*N*K 可能为0，导致除零。这里k至少为1.
    double gflops = 0.0;
    if (avg_time_sec > 0 && m > 0 && n > 0 && k > 0) { // 避免除以零或负的GFLOPS
        gflops = (2.0 * (double)m * (double)n * (double)k) / avg_time_sec / 1e9;
    }
    printf("   平均耗时: %.6f 秒, GFLOPS: %.2f\n", avg_time_sec, gflops);

    // 如果提供了参考结果 C_ref，则进行验证
    if (C_ref) {
        int diff_r, diff_c;
        double max_diff = verify_result(C_ref, C_test, m, n, &diff_r, &diff_c);
        if (diff_r != -1 && max_diff > 1e-9) { // 仅当差异显著时打印细节
            printf("   验证: 结果最大差异 = %e 在位置 C(%d,%d)\n", max_diff, diff_r, diff_c);
            // 可选: 打印参考值和计算值以进行更详细的调试
            // printf("        参考值 C_ref(%d,%d): %f\n", diff_r, diff_c, C_idx(C_ref, diff_r, diff_c, n));
            // printf("        计算值 C_test(%d,%d): %f\n", diff_r, diff_c, C_idx(C_test, diff_r, diff_c, n));
        } else if (max_diff <= 1e-9) {
             printf("   验证: 结果与参考值一致 (最大差异 <= 1e-9)。\n");
        } else { // diff_r == -1 (通常意味着 C_ref 和 C_test 完全相同)
             printf("   验证: 结果与参考值完全一致。\n");
        }
    }
    printf("\n"); // 在不同算法的输出之间添加空行
}


// --- 主函数 ---
int main() {
    // 初始化随机数生成器种子，确保每次运行产生不同的随机数序列
    srand(time(NULL));

    printf("矩阵乘法性能测试 (交互式输入维度)\n");
    printf("=====================================================\n");
    printf("每个算法将针对每个维度运行 %d 次并计算平均性能。\n", NUM_RUNS);
    printf("分块优化使用的块大小为: %d\n", DEFAULT_BLOCK_SIZE);
    printf("微内核优化当前使用的是其 4x4 标量模拟版本。\n");
    printf("=====================================================\n\n");

    char more_tests = 'y';
    while (more_tests == 'y' || more_tests == 'Y') {
        int m, n, k;
        printf("请输入矩阵维度 M N K (以空格分隔, 例如: 100 100 100): ");
        if (scanf("%d %d %d", &m, &n, &k) != 3) {
            printf("输入错误，请输入三个整数值。\n");
            // 清除输入缓冲区以防无限循环
            while (getchar() != '\n'); 
            continue; // 跳过本次迭代，重新提示输入
        }

        if (m <= 0 || n <= 0 || k <= 0) {
            printf("错误：矩阵维度 M, N, K 必须是正整数。\n");
            continue;
        }

        printf("\n开始测试维度: M=%d, N=%d, K=%d\n", m, n, k);
        printf("-----------------------------------------------------\n");

        // 分配矩阵内存
        double* A      = allocate_matrix(m, k);
        double* B      = allocate_matrix(k, n);
        double* C_ref  = allocate_matrix(m, n); // 用于存储朴素算法的结果，作为参考
        double* C_test = allocate_matrix(m, n); // 用于测试各种优化算法

        // 使用随机值填充输入矩阵 A 和 B
        fill_random_matrix(A, m, k);
        fill_random_matrix(B, k, n);

        // 1. 运行朴素 ijk 实现 (并将结果存储在 C_ref 中作为后续验证的基准)
        run_and_time_gemm("0. 朴素 ijk 实现", 
                          gemm_naive_ijk, NULL, 
                          m, n, k, A, B, C_ref, 
                          NULL, 0); // 第一个运行，C_ref 为 NULL (不进行自我验证)

        // 2. 运行循环顺序优化 ikj (gemm_opt_ikj 来自 gemm_optimized_loops.c)
        run_and_time_gemm("1. 循环顺序优化 ikj",
                          gemm_opt_ikj, NULL,
                          m, n, k, A, B, C_test,
                          C_ref, 0); // 使用 C_ref 进行验证

        // 3. 运行分块优化 (gemm_blocking)
        run_and_time_gemm("2. 分块优化",
                          NULL, gemm_blocking, // 注意这里 func 为 NULL, blocking_func 被使用
                          m, n, k, A, B, C_test,
                          C_ref, 1); // use_blocking_func = 1

        // 4. 运行微内核 AVX2 优化封装版本 (gemm_microkernel_avx2_wrapper 来自 gemm_microkernel.c)
        //    当前此函数内部调用的是 gemm_kernel_4x4_scalar 进行正确性验证。
        run_and_time_gemm("3. 微内核 AVX2 (4x4 标量模拟)",
                          gemm_microkernel_avx2_wrapper, NULL,
                          m, n, k, A, B, C_test,
                          C_ref, 0);
        
        // 释放为当前维度分配的矩阵内存
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_ref);
        free_matrix(C_test);

        printf("维度 M=%d, N=%d, K=%d 测试完成。\n", m, n, k);
        printf("=====================================================\n");

        printf("是否要测试其他维度? (y/n): ");
        // 清除上一个 scanf 留下的换行符
        while (getchar() != '\n'); 
        scanf(" %c", &more_tests); // 注意 scanf(" %c") 中的空格，它可以消耗掉之前输入留下的任何空白字符
        // 再次清除输入缓冲区，以防用户输入多个字符
        while (getchar() != '\n'); 
        printf("\n");
    }

    printf("所有测试已完成。\n");
    return 0;
} 