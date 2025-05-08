# Makefile for GEMM Performance Tester

# 编译器和标志
CC = gcc
# CFLAGS: 编译 C 文件时的选项
# -O3:        启用高级别优化
# -Wall:      启用所有核心警告 (all warnings)
# -Wextra:    启用一些额外的警告
# -pedantic:  严格遵循 C 标准，并对不符合标准的用法发出警告
# -std=c11:   指定使用 C11 标准 (如果您的编译器支持更新的，如c17, c2x也可以)
# -mavx2:     启用 AVX2 指令集支持 (即使当前是模拟，为将来做准备)
# -g:         包含调试信息 (可选, 如果需要gdb调试可以加上)
CFLAGS = -O3 -Wall -Wextra -pedantic -std=c11 -mavx2

# LDFLAGS: 链接时的选项
# -lm: 链接数学库 (因为使用了 fabs 等)
LDFLAGS = -lm

# 可执行文件名
TARGET = gemm_tester

# 源文件: 明确列出所有需要编译的 .c 文件
# 避免使用 wildcard 以防止意外包含不需要的文件
SRCS = main.c gemm_naive.c gemm_optimized_loops.c gemm_blocking.c gemm_microkernel.c

# 对象文件: 将 .c 文件名替换为 .o 文件名
OBJS = $(SRCS:.c=.o)

# 默认目标: 当只输入 'make' 时执行
all: $(TARGET)

# 链接规则: 如何从对象文件生成可执行文件
# $@: 目标文件名 (即 $(TARGET))
# $^: 所有依赖项的文件名 (即 $(OBJS))
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "链接完成: $@"

# 编译规则: 如何从 .c 文件生成 .o 对象文件
# $<: 第一个依赖项的文件名 (即对应的 .c 文件)
# -c: 只编译，不链接
# 每个.o文件依赖于其对应的.c文件以及common.h
%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "已编译: $< -> $@"

# 清理规则: 删除生成的文件
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "清理完成。"

# 将 'all' 和 'clean' 声明为伪目标，因为它们不代表实际的文件名
.PHONY: all clean
