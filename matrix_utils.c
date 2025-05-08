#include "common.h" 

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
        perror("错误：无法分配矩阵内存 (来自 matrix_utils.c)");
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