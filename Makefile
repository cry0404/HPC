# Makefile for GEMM Performance Tester

# 编译器和标志
CC = gcc
# CFLAGS: 编译 C 文件时的选项
# -O3:        启用高级别优化
# -Wall:      启用所有核心警告 (all warnings)
# -Wextra:    启用一些额外的警告
# -pedantic:  严格遵循 C 标准，并对不符合标准的用法发出警告
# -std=c11:   指定使用 C11 标准 (如果您的编译器支持更新的，如c17, c2x也可以)
# -mavx2:     启用 AVX2 指令集支持
# -mfma:      显式启用 FMA 指令集支持 (通常被 -mavx2 包含，但有时需要明确指出)
# -g:         包含调试信息 (可选, 如果需要gdb调试可以加上)
CFLAGS = -O3 -Wall -Wextra -pedantic -std=c11 -mavx2 -mfma

# LDFLAGS: 链接时的选项
# -lm: 链接数学库 (因为使用了 fabs 等)
LDFLAGS = -lm

# 可执行文件名
TARGET_TESTER = gemm_tester
TARGET_RUNNER = gemm_runner

# --- 源文件列表 ---
# 通用工具函数源文件
UTIL_SRCS = matrix_utils.c

# GEMM 算法实现源文件 (被两个目标程序共享)
GEMM_ALGO_SRCS = gemm_naive.c gemm_optimized_loops.c gemm_blocking.c gemm_microkernel.c gemm_packing.c

# gemm_tester 源文件
TESTER_SRCS = main.c

# gemm_runner 源文件
RUNNER_SRCS = gemm_runner.c

# --- 对象文件列表 ---
UTIL_OBJS = $(UTIL_SRCS:.c=.o)
GEMM_ALGO_OBJS = $(GEMM_ALGO_SRCS:.c=.o)
TESTER_OBJS = $(TESTER_SRCS:.c=.o)
RUNNER_OBJS = $(RUNNER_SRCS:.c=.o)

# 默认目标: 构建所有可执行文件
all: $(TARGET_TESTER) $(TARGET_RUNNER)

# --- 链接规则 ---
# 链接 gemm_tester
$(TARGET_TESTER): $(TESTER_OBJS) $(GEMM_ALGO_OBJS) $(UTIL_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "链接完成: $@ (测试器)"

# 链接 gemm_runner
$(TARGET_RUNNER): $(RUNNER_OBJS) $(GEMM_ALGO_OBJS) $(UTIL_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "链接完成: $@ (运行器)"

# --- 通用编译规则 --- 
# 如何从 .c 文件生成 .o 对象文件
# 每个 .o 文件依赖于其对应的 .c 文件以及 common.h
%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "已编译: $< -> $@"

# --- 清理规则 ---
clean:
	rm -f $(TESTER_OBJS) $(RUNNER_OBJS) $(GEMM_ALGO_OBJS) $(UTIL_OBJS) $(TARGET_TESTER) $(TARGET_RUNNER)
	@echo "清理完成。"

# 将 'all' 和 'clean' 声明为伪目标
.PHONY: all clean
