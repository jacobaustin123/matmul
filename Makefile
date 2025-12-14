CC = clang
CFLAGS = -O2 -Wall -fno-vectorize -fno-slp-vectorize
FRAMEWORKS = -framework Accelerate

TARGET = matmul_test

all: $(TARGET)

$(TARGET): main.c matmul.s
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ main.c matmul.s

clean:
	rm -f $(TARGET)

.PHONY: all clean
