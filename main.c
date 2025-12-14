#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// Assembly functions
extern void neon_matmul_4x4(float *C, const float *A, const float *B);

// Scalar reference implementation
void scalar_matmul_4x4(float *C, const float *A, const float *B) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += A[i * 4 + k] * B[k * 4 + j];
            }
            C[i * 4 + j] = sum;
        }
    }
}

void print_matrix(const char *name, const float *M, int N) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) {
            printf("%8.2f", M[i * N + j]);
        }
        printf(" ]\n");
    }
}

int matrices_equal(const float *A, const float *B, int N, float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i * N + j] - B[i * N + j] > tolerance) return 0;
        }
    }
    return 1;
}

int subtract(float * C, const float *A, const float *B, int N, float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = A[i * N + j] - B[i * N + j];
        }
    }
    return 1;
}

// For benchmarking: larger matrix multiply using 4x4 blocks (tiled)
void scalar_matmul_large(float *C, const float *A, const float *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Tiled matmul using our NEON 4x4 kernel
void neon_matmul_tiled(float *C, const float *A, const float *B, int N) {
    // Zero out C first
    memset(C, 0, N * N * sizeof(float));
    
    // Temporary 4x4 blocks
    float A_block[16], B_block[16], C_block[16];
    
    // Tile over the matrices
    for (int i = 0; i < N; i += 4) {
        for (int j = 0; j < N; j += 4) {
            // Zero the C block accumulator
            memset(C_block, 0, sizeof(C_block));
            
            for (int k = 0; k < N; k += 4) {
                // Extract 4x4 block of A starting at (i, k)
                for (int bi = 0; bi < 4; bi++) {
                    for (int bj = 0; bj < 4; bj++) {
                        A_block[bi * 4 + bj] = A[(i + bi) * N + (k + bj)];
                    }
                }
                
                // Extract 4x4 block of B starting at (k, j)
                for (int bi = 0; bi < 4; bi++) {
                    for (int bj = 0; bj < 4; bj++) {
                        B_block[bi * 4 + bj] = B[(k + bi) * N + (j + bj)];
                    }
                }
                
                // Multiply blocks and accumulate
                float temp[16];
                neon_matmul_4x4(temp, A_block, B_block);
                for (int idx = 0; idx < 16; idx++) {
                    C_block[idx] += temp[idx];
                }
            }
            
            // Write C block back to C
            for (int bi = 0; bi < 4; bi++) {
                for (int bj = 0; bj < 4; bj++) {
                    C[(i + bi) * N + (j + bj)] = C_block[bi * 4 + bj];
                }
            }
        }
    }
}

// Accelerate framework wrapper
void accelerate_matmul(float *C, const float *A, const float *B, int N) {
    // cblas_sgemm: C = alpha * A * B + beta * C
    // CblasRowMajor: row-major storage
    // CblasNoTrans: don't transpose A or B
    // M, N, K: dimensions (M x K) * (K x N) = (M x N)
    // alpha = 1.0, beta = 0.0: C = A * B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,           // M, N, K
                1.0f,              // alpha
                A, N,              // A, lda
                B, N,              // B, ldb
                0.0f,              // beta
                C, N);             // C, ldc
}

int main() {
    // Test matrices
    float A[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float B[16] = {
        17, 18, 19, 20,
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
    };
    
    float C_scalar[16], C_neon[16], C_accel[16];
    
    printf("=== 4x4 Matrix Multiply Test ===\n\n");
    
    print_matrix("A", A, 4);
    printf("\n");
    print_matrix("B", B, 4);
    printf("\n");
    
    // Compute reference
    scalar_matmul_4x4(C_scalar, A, B);
    print_matrix("C (scalar)", C_scalar, 4);
    printf("\n");
    
    // Compute NEON version
    neon_matmul_4x4(C_neon, A, B);
    print_matrix("C (NEON)", C_neon, 4);
    printf("\n");

    accelerate_matmul(C_accel, A, B, 4);
    print_matrix("C (Accel)", C_neon, 4);
    printf("\n");

    if (matrices_equal(C_scalar, C_neon, 4, 0.001f)) {
        printf("✓ Results match (Neon)!\n\n");
    } else {
        printf("✗ Results differ (Neon)!\n\n");
    }
    
    if (matrices_equal(C_scalar, C_accel, 4, 0.001f)) {
        printf("✓ Results match (Accel)!\n\n");
    } else {
        printf("✗ Results differ (Accel)!\n\n");
    }
    
    // Benchmark
    printf("=== Benchmark ===\n\n");
    
    const int ITERATIONS = 10000000;
    clock_t start, end;
    
    // Benchmark scalar 4x4
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        scalar_matmul_4x4(C_scalar, A, B);
    }
    end = clock();
    double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Benchmark NEON 4x4
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        neon_matmul_4x4(C_neon, A, B);
    }
    end = clock();
    double neon_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Benchmark Accelerate
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        accelerate_matmul(C_accel, A, B, 4);
    }
    end = clock();
    double accel_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("4x4 matmul (%d iterations):\n", ITERATIONS);
    printf("  Scalar: %.3f seconds\n", scalar_time);
    printf("  NEON:   %.3f seconds\n", neon_time);
    printf("  Accel:   %.3f seconds\n", accel_time);
    printf("  NEON Speedup: %.2fx\n\n", scalar_time / neon_time);
    printf("  Accel Speedup: %.2fx\n\n", scalar_time / accel_time);

    // Larger matrix benchmark
    printf("=== Larger Matrix Benchmark (64x64) ===\n\n");
    
    const int N = 64;
    float *bigA = malloc(N * N * sizeof(float));
    float *bigB = malloc(N * N * sizeof(float));
    float *bigC_scalar = malloc(N * N * sizeof(float));
    float *bigC_neon = malloc(N * N * sizeof(float));
    float *bigC_accel = malloc(N * N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        bigA[i] = (float)(rand() % 100) / 10.0f;
        bigB[i] = (float)(rand() % 100) / 10.0f;
    }
    
    const int LARGE_ITERS = 1000;
    
    // Benchmark scalar
    start = clock();
    for (int i = 0; i < LARGE_ITERS; i++) {
        scalar_matmul_large(bigC_scalar, bigA, bigB, N);
    }
    end = clock();
    scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Benchmark tiled NEON
    start = clock();
    for (int i = 0; i < LARGE_ITERS; i++) {
        neon_matmul_tiled(bigC_neon, bigA, bigB, N);
    }
    end = clock();
    neon_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Benchmark Accelerate
    start = clock();
    for (int i = 0; i < LARGE_ITERS; i++) {
        accelerate_matmul(bigC_accel, bigA, bigB, N);
    }
    end = clock();
    accel_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("64x64 matmul (%d iterations):\n", LARGE_ITERS);
    printf("  Scalar:       %.3f seconds\n", scalar_time);
    printf("  NEON tiled:   %.3f seconds\n", neon_time);
    printf("  Accelerate:   %.3f seconds\n", accel_time);
    printf("  NEON speedup vs scalar:   %.2fx\n", scalar_time / neon_time);
    printf("  Accel speedup vs scalar:  %.2fx\n", scalar_time / accel_time);
    printf("  Accel speedup vs NEON:    %.2fx\n", neon_time / accel_time);
    printf("\n");

    if (matrices_equal(bigC_scalar, bigC_neon, N, 0.001f)) {
        printf("✓ Results match (NEON)!\n\n");
    } else {
        printf("✗ Results differ (NEON)!\n\n");
    }
    
    if (matrices_equal(bigC_scalar, bigC_accel, N, 0.001f)) {
        printf("✓ Results match (Accel)!\n\n");
    } else {
        printf("✗ Results differ (Accell)!\n\n");
    }
    
    // Even larger test
    printf("\n=== Larger Matrix Benchmark (512x512) ===\n\n");
    
    const int N2 = 512;
    float *hugeA = malloc(N2 * N2 * sizeof(float));
    float *hugeB = malloc(N2 * N2 * sizeof(float));
    float *hugeC_scalar = malloc(N2 * N2 * sizeof(float));
    float *hugeC_neon = malloc(N2 * N2 * sizeof(float));
    float *hugeC_accel = malloc(N2 * N2 * sizeof(float));
    
    for (int i = 0; i < N2 * N2; i++) {
        hugeA[i] = (float)(rand() % 100) / 100.0f;
        hugeB[i] = (float)(rand() % 100) / 100.0f;
    }
    
    const int HUGE_ITERS = 10;
    
    start = clock();
    for (int i = 0; i < HUGE_ITERS; i++) {
        scalar_matmul_large(hugeC_scalar, hugeA, hugeB, N2);
    }
    end = clock();
    scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    for (int i = 0; i < HUGE_ITERS; i++) {
        neon_matmul_tiled(hugeC_neon, hugeA, hugeB, N2);
    }
    end = clock();
    neon_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    for (int i = 0; i < HUGE_ITERS; i++) {
        accelerate_matmul(hugeC_accel, hugeA, hugeB, N2);
    }
    end = clock();
    accel_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("256x256 matmul (%d iterations):\n", HUGE_ITERS);
    printf("  Scalar:       %.3f seconds\n", scalar_time);
    printf("  NEON tiled:   %.3f seconds\n", neon_time);
    printf("  Accelerate:   %.3f seconds\n", accel_time);
    printf("  NEON speedup vs scalar:   %.2fx\n", scalar_time / neon_time);
    printf("  Accel speedup vs scalar:  %.2fx\n", scalar_time / accel_time);
    printf("  Accel speedup vs NEON:    %.2fx\n", neon_time / accel_time);
    printf("\n");

    if (matrices_equal(hugeC_scalar, hugeC_neon, N2, 0.001f)) {
        printf("✓ Results match (NEON)!\n\n");
    } else {
        printf("✗ Results differ (NEON)!\n\n");
    }
    
    if (matrices_equal(hugeC_scalar, hugeC_accel, N2, 0.001f)) {
        printf("✓ Results match (Accel)!\n\n");
    } else {
        printf("✗ Results differ (Accell)!\n\n");
    }
    
    // print_matrix("scalar", hugeC_scalar, 4);
    // print_matrix("neon", hugeC_neon, 4);
    
    free(bigA);
    free(bigB);
    free(bigC_scalar);
    free(bigC_neon);
    free(bigC_accel);
    free(hugeA);
    free(hugeB);
    free(hugeC_scalar);
    free(hugeC_neon);
    free(hugeC_accel);
    
    return 0;
}
