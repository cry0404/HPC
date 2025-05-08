#include "common.h"
#include <string.h> // 用于 strcmp 函数
#include <stdio.h>  // 用于 printf, fprintf, stderr 标准输入输出函数
#include <stdlib.h> // 用于 atoi (字符串转整数), exit, EXIT_FAILURE, malloc, free 内存管理函数
#include <time.h>   // 用于 clock_gettime (高精度计时), time (获取时间), srand (随机数种子)
#include <unistd.h> // 用于 getopt 函数解析命令行选项

// 如果未另外指定，分块算法使用的默认块大小
#define RUNNER_DEFAULT_BLOCK_SIZE 16
// 定义运行次数。对于 perf 分析，单次运行通常足够；或者用3次进行简单的平均。
#define RUNNER_NUM_RUNS 1 

// 打印 gemm_runner 用法说明的静态辅助函数
static void print_runner_usage(const char* prog_name) {
    fprintf(stderr, "用法: %s <算法> -m <M值> -n <N值> -k <K值> [-h]\n", prog_name);
    fprintf(stderr, "选项:\n");
    fprintf(stderr, "  -m <数值>   矩阵维度 M (必需)\n");
    fprintf(stderr, "  -n <数值>   矩阵维度 N (必需)\n");
    fprintf(stderr, "  -k <数值>   矩阵维度 K (必需)\n");
    fprintf(stderr, "  -h          显示此帮助信息并退出\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "可用的 <算法> (必需的第一个非选项参数):\n");
    fprintf(stderr, "  naive       - 朴素的 ijk 实现\n");
    fprintf(stderr, "  ikj         - 循环顺序优化 ikj\n");
    fprintf(stderr, "  blocking    - 分块优化 (使用默认块大小 %d)\n", RUNNER_DEFAULT_BLOCK_SIZE);
    fprintf(stderr, "  micro       - 微内核 AVX2 (4x4 真实SIMD, 无外层Packing)\n");
    fprintf(stderr, "  packing     - Packing + AVX2 微内核\n");
    exit(EXIT_FAILURE); // 打印用法后退出程序，表示失败
}

// 主函数入口点
int main(int argc, char *argv[]) {
    int m = -1, n = -1, k = -1; // 初始化为无效值，表示必须由用户提供
    int opt;
    const char* algo_name = NULL;

    // 使用 getopt 解析短选项 m:, n:, k:, h
    // 冒号表示选项后面需要一个参数 (optarg)
    while ((opt = getopt(argc, argv, "m:n:k:h")) != -1) {
        switch (opt) {
            case 'm':
                m = atoi(optarg); // optarg 指向选项的参数
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            case 'h': // 处理帮助选项
                print_runner_usage(argv[0]);
                break;
            case '?': // 处理无效选项或缺少参数的情况
                // getopt 已经打印了错误消息
                fprintf(stderr, "尝试 '%s -h' 获取更多信息。\n", argv[0]);
                return EXIT_FAILURE;
            default:
                // 理论上不应到达这里
                abort();
        }
    }

    // getopt 处理完所有选项后，optind 是第一个非选项参数的索引
    // 我们期望算法名称是第一个非选项参数
    if (optind >= argc) { // 检查是否提供了算法名称
        fprintf(stderr, "错误: 必须提供算法名称。\n");
        print_runner_usage(argv[0]);
    } else {
        algo_name = argv[optind];
        // (可选) 检查后面是否还有其他非选项参数
        if (optind + 1 < argc) {
             fprintf(stderr, "警告: 忽略了多余的非选项参数 '%s'...\n", argv[optind + 1]);
        }
    }

    // 校验必需的维度参数是否已提供且有效
    if (m <= 0 || n <= 0 || k <= 0) {
        fprintf(stderr, "错误: 必须通过 -m, -n, -k 提供有效的正整数维度。\n");
        print_runner_usage(argv[0]);
    }

    printf("运行算法: %s, 维度: M=%d, N=%d, K=%d\n", algo_name, m, n, k);

    // 分配矩阵 A(m,k), B(k,n), C(m,n)
    double* A = allocate_matrix(m, k); 
    double* B = allocate_matrix(k, n);
    double* C = allocate_matrix(m, n);
    if (!A || !B || !C) {
        fprintf(stderr, "错误: 矩阵分配失败。\n");
        if (A) free_matrix(A);
        if (B) free_matrix(B);
        if (C) free_matrix(C);
        return EXIT_FAILURE;
    }

    // 初始化矩阵
    srand(time(NULL)); 
    fill_random_matrix(A, m, k); 
    fill_random_matrix(B, k, n);

    double total_time_sec = 0.0; 
    struct timespec start_time, end_time; 

    // 根据算法名称选择要执行的GEMM函数
    void (*gemm_function_to_run)(int, int, int, const double*, const double*, double*) = NULL;
    int is_blocking_algo = 0; 

    if (strcmp(algo_name, "naive") == 0) {
        gemm_function_to_run = gemm_naive_ijk;
    } else if (strcmp(algo_name, "ikj") == 0) {
        gemm_function_to_run = gemm_opt_ikj;
    } else if (strcmp(algo_name, "blocking") == 0) {
        is_blocking_algo = 1;
    } else if (strcmp(algo_name, "micro") == 0) {
        gemm_function_to_run = gemm_microkernel_avx2_wrapper;
    } else if (strcmp(algo_name, "packing") == 0) {
        gemm_function_to_run = gemm_packed_avx2;
    } else {
        fprintf(stderr, "错误: 未知的算法名称 '%s'\n", algo_name);
        print_runner_usage(argv[0]);
    }

    printf("准备运行 %d 次...\n", RUNNER_NUM_RUNS);

    for (int run = 0; run < RUNNER_NUM_RUNS; ++run) {
        zero_matrix(C, m, n); 
        
        clock_gettime(CLOCK_MONOTONIC, &start_time); 

        if (is_blocking_algo) {
            gemm_blocking(m, n, k, A, B, C, RUNNER_DEFAULT_BLOCK_SIZE);
        } else if (gemm_function_to_run) {
            gemm_function_to_run(m, n, k, A, B, C);
        } else {
            fprintf(stderr, "内部错误：没有选择有效的GEMM函数。\n");
            exit(EXIT_FAILURE);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end_time); 
        double time_taken_sec = (end_time.tv_sec - start_time.tv_sec) +
                                (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        total_time_sec += time_taken_sec; 
        if (RUNNER_NUM_RUNS > 1) { 
            printf("   迭代 %d 耗时: %.6f 秒\n", run + 1, time_taken_sec);
        }
    }

    double avg_time_sec = total_time_sec / RUNNER_NUM_RUNS; 
    double gflops = 0.0; 
    if (avg_time_sec > 1e-9 && m > 0 && n > 0 && k > 0) { 
        gflops = (2.0 * (double)m * (double)n * (double)k) / avg_time_sec / 1e9;
    }

    printf("-----------------------------------------------------\n");
    if (RUNNER_NUM_RUNS > 1) {
        printf("平均耗时: %.6f 秒\n", avg_time_sec);
    }
    printf("总耗时  : %.6f 秒 (对于 %d 次运行)\n", total_time_sec, RUNNER_NUM_RUNS);
    printf("GFLOPS    : %.2f\n", gflops);
    printf("-----------------------------------------------------\n");

    // 释放分配的矩阵内存
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    printf("算法 %s 执行完毕。\n", algo_name);
    return 0; // 程序正常结束
} 