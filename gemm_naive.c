#include "common.h" // 引入我们之前定义的通用头文件

// 函数名: gemm_naive_ijk
// 功能: 实现最基础的 ijk 循环顺序的矩阵乘法 C = A * B
//       (更准确地说，这里实现的是 C += A*B 的累加形式，
//        所以调用前需要确保 C 矩阵被初始化为0)
// 参数:
//   m: C 和 A 的行数
//   n: C 和 B 的列数
//   k: A 的列数 / B 的行数 (公共维度)
//   A: 指向矩阵 A 的指针
//   B: 指向矩阵 B 的指针
//   C: 指向结果矩阵 C 的指针
void gemm_naive_ijk(int m, int n, int k, double* A, double* B, double* C) {
    // 外层循环: 遍历结果矩阵 C 的每一行
    for (int i = 0; i < m; ++i) {
        // 中层循环: 遍历结果矩阵 C 的每一列
        for (int j = 0; j < n; ++j) {
            double sum_val = 0.0; // 临时变量，用于累加 C(i,j) 的值
                                 // 存放在寄存器中可以提高效率
            // 内层循环: 计算点积，即 A 的第 i 行与 B 的第 j 列的乘积累加
            for (int p = 0; p < k; ++p) {
                // C(i,j) = sum_{p=0}^{k-1} A(i,p) * B(p,j)
                // A_idx 和 B_idx 是 common.h 中定义的宏，用于访问矩阵元素
                // A_idx(A, i, p, k) 访问 A[i][p]
                // B_idx(B, p, j, n) 访问 B[p][j]
                sum_val += A_idx(A, i, p, k) * B_idx(B, p, j, n);
            }
            // 将计算得到的和赋值给 C(i,j)
            // 如果是累加模式 C += A*B, 应该是 C_idx(C, i, j, n) += sum_val;
            // 但为了与后续优化版本一致（它们通常从0开始累加），这里直接赋值。
            // 调用者 (main.c) 需要确保 C 在调用前被清零。
            C_idx(C, i, j, n) = sum_val;
        }
    }
}