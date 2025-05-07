#include "common.h" // 引入我们的通用头文件

// --- 声明我们稍后会实现的各种矩阵乘法函数 ---
void gemm_naive_ijk(int m, int n, int k, double* A, double* B, double* C);
void gemm_opt_ikj(int m, int n, int k, double* A, double* B, double* C);
void gemm_blocked(int m, int n, int k, double* A, double* B, double* C, int block_size); 
void gemm_microkernel_avx2_wrapper(int m, int n, int k, double* A, double* B, double* C);

// 定义测试重复次数
#define REPEAT_TIMES 3  // 每个算法重复测试的次数

int main(int argc, char** argv) {
    int N_size = 40; // 默认矩阵大小 M=N_size, K=N_size 
    if (argc > 1) {
        N_size = atoi(argv[1]); // 如果命令行提供了参数，则使用该参数作为矩阵大小
    }

    // 为了简化 AVX2 微内核的边界演示，确保 N_size 是 4 的倍数
    if (N_size % 4 != 0) {
        int N_original = N_size; 
        N_size = (N_size / 4) * 4;
        if (N_size == 0 && N_original > 0) N_size = 4;
        else if (N_size == 0 && N_original == 0) { printf("错误: 矩阵大小为0。\n"); return 1; }
        printf("提示：矩阵大小从 %d 调整为 %d (4的倍数) 以适配微内核。\n", N_original, N_size);
    }
    if (N_size <= 0) { printf("错误: 矩阵大小必须为正。\n"); return 1; }

    int m = N_size, n = N_size, k = N_size; // 方阵

    printf("矩阵乘法测试: C(%dx%d) = A(%dx%d) * B(%dx%d)\n", m, n, m, k, k, n);
    printf("每个算法将运行 %d 次取平均值\n", REPEAT_TIMES);

    // 使用 aligned_alloc 分配对齐内存，对 SIMD 有益
    size_t alignment = 64; // 64字节对齐，对 AVX-512 友好，也适用于 AVX2
    double* matrix_A = (double*)aligned_alloc(alignment, m * k * sizeof(double)); 
    double* matrix_B = (double*)aligned_alloc(alignment, k * n * sizeof(double));
    double* matrix_C_ref = (double*)aligned_alloc(alignment, m * n * sizeof(double)); // 参考结果 (朴素算法)
    double* matrix_C_opt = (double*)aligned_alloc(alignment, m * n * sizeof(double)); // 优化算法结果

    if (!matrix_A || !matrix_B || !matrix_C_ref || !matrix_C_opt) {
        perror("内存分配失败");
        if(matrix_A) free(matrix_A); 
        if(matrix_B) free(matrix_B); 
        if(matrix_C_ref) free(matrix_C_ref); 
        if(matrix_C_opt) free(matrix_C_opt);
        return 1;
    }

    srand(0); // 固定随机数种子，确保每次运行的输入数据一致
    init_matrix(m, k, matrix_A);
    init_matrix(k, n, matrix_B);

    double time_start, time_end, gflops; // 变量名用英文
    double total_time; // 用于累计多次运行的总时间
    long long num_operations = 2LL * m * n * k; // 估算的浮点运算次数 (变量名用英文)

    printf("\n--- 运行各版本优化 (N=%d) ---\n", N_size);

    // 0. 朴素 ijk (基准)
    printf("\n0. 朴素 ijk 实现:\n");
    // 缓存预热
    memset(matrix_C_ref, 0, m * n * sizeof(double));
    gemm_naive_ijk(m, n, k, matrix_A, matrix_B, matrix_C_ref);
    
    // 正式测量
    total_time = 0.0;
    for (int iter = 0; iter < REPEAT_TIMES; iter++) {
        memset(matrix_C_ref, 0, m * n * sizeof(double)); // 清零结果矩阵
        time_start = get_time();
        gemm_naive_ijk(m, n, k, matrix_A, matrix_B, matrix_C_ref);
        time_end = get_time();
        total_time += (time_end - time_start);
        printf("   迭代 %d 耗时: %.6f 秒\n", iter + 1, time_end - time_start);
    }
    gflops = (num_operations * REPEAT_TIMES * 1e-9) / total_time;
    printf("   平均耗时: %.6f 秒, GFLOPS: %.2f\n", total_time / REPEAT_TIMES, gflops);

    // 1. 循环顺序优化 ikj
    printf("\n1. 循环顺序优化 ikj:\n");
    // 缓存预热
    memset(matrix_C_opt, 0, m * n * sizeof(double));
    gemm_opt_ikj(m, n, k, matrix_A, matrix_B, matrix_C_opt);
    
    // 正式测量
    total_time = 0.0;
    for (int iter = 0; iter < REPEAT_TIMES; iter++) {
        memset(matrix_C_opt, 0, m * n * sizeof(double));
        time_start = get_time();
        gemm_opt_ikj(m, n, k, matrix_A, matrix_B, matrix_C_opt);
        time_end = get_time();
        total_time += (time_end - time_start);
        printf("   迭代 %d 耗时: %.6f 秒\n", iter + 1, time_end - time_start);
    }
    gflops = (num_operations * REPEAT_TIMES * 1e-9) / total_time;
    printf("   平均耗时: %.6f 秒, GFLOPS: %.2f\n", total_time / REPEAT_TIMES, gflops);
    verify_gemm(m, n, matrix_C_opt, matrix_C_ref);

    // 2. 分块优化
    int block_size_val = 16; // 可以调整此参数进行实验 (变量名用英文)
    if (block_size_val > N_size) block_size_val = N_size;
    if (N_size % 4 == 0 && block_size_val % 4 != 0 && block_size_val > 4) block_size_val = (block_size_val/4)*4;
    if (block_size_val == 0) block_size_val = 4;

    printf("\n2. 分块优化 (块大小 = %d):\n", block_size_val);
    // 缓存预热
    memset(matrix_C_opt, 0, m * n * sizeof(double));
    gemm_blocked(m, n, k, matrix_A, matrix_B, matrix_C_opt, block_size_val);
    
    // 正式测量
    total_time = 0.0;
    for (int iter = 0; iter < REPEAT_TIMES; iter++) {
        memset(matrix_C_opt, 0, m * n * sizeof(double));
        time_start = get_time();
        gemm_blocked(m, n, k, matrix_A, matrix_B, matrix_C_opt, block_size_val);
        time_end = get_time();
        total_time += (time_end - time_start);
        printf("   迭代 %d 耗时: %.6f 秒\n", iter + 1, time_end - time_start);
    }
    gflops = (num_operations * REPEAT_TIMES * 1e-9) / total_time;
    printf("   平均耗时: %.6f 秒, GFLOPS: %.2f\n", total_time / REPEAT_TIMES, gflops);
    verify_gemm(m, n, matrix_C_opt, matrix_C_ref);

    // 3. 微内核 AVX2 优化
    // (main函数开头已确保N_size是4的倍数)
    printf("\n3. 微内核 AVX2 优化 (4x4):\n");
    // 缓存预热
    memset(matrix_C_opt, 0, m * n * sizeof(double));
    gemm_microkernel_avx2_wrapper(m, n, k, matrix_A, matrix_B, matrix_C_opt);
    
    // 正式测量
    total_time = 0.0;
    for (int iter = 0; iter < REPEAT_TIMES; iter++) {
        memset(matrix_C_opt, 0, m * n * sizeof(double));
        time_start = get_time();
        gemm_microkernel_avx2_wrapper(m, n, k, matrix_A, matrix_B, matrix_C_opt);
        time_end = get_time();
        total_time += (time_end - time_start);
        printf("   迭代 %d 耗时: %.6f 秒\n", iter + 1, time_end - time_start);
    }
    gflops = (num_operations * REPEAT_TIMES * 1e-9) / total_time;
    printf("   平均耗时: %.6f 秒, GFLOPS: %.2f\n", total_time / REPEAT_TIMES, gflops);
    verify_gemm(m, n, matrix_C_opt, matrix_C_ref);

    // 释放内存
    free(matrix_A);
    free(matrix_B);
    free(matrix_C_ref);
    free(matrix_C_opt);

    return 0;
}