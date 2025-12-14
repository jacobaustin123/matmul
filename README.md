# ARM64 NEON Matrix Multiplication

A benchmark comparing matrix multiplication implementations on Apple Silicon:

- **Scalar**: Naive triple-loop C implementation
- **NEON**: Hand-written ARM64 assembly using NEON SIMD instructions
- **Accelerate**: Apple's optimized BLAS via `cblas_sgemm`

## Building

```bash
make
```

Requires macOS with Apple Silicon (or ARM64) and Xcode command line tools.

## Running

```bash
./matmul_test
```

Benchmarks 4x4, 64x64, and 512x512 matrix multiplications.

## Implementation Details

The NEON kernel (`matmul.s`) performs 4x4 matrix multiplication using:
- `ld1` for loading matrix rows into vector registers
- `fmul`/`fmla` for fused multiply-accumulate operations
- Each output row is computed as a linear combination of B's rows, weighted by the corresponding A row elements

The tiled implementation extends this to larger matrices by blocking into 4x4 tiles.

## License

MIT
