// matmul.s - Matrix multiplication using NEON SIMD
// NEON 4x4 matrix multiply: C = A * B

.global _neon_matmul_4x4
.align 4

_neon_matmul_4x4:
    // x0 = C, x1 = A, x2 = B
    
    // Load all of matrix B into v16-v19 (rows)
    ld1     {v16.4s}, [x2], #16      // B row 0
    ld1     {v17.4s}, [x2], #16      // B row 1
    ld1     {v18.4s}, [x2], #16      // B row 2
    ld1     {v19.4s}, [x2], #16      // B row 3

    // Process each row of A
    // Row 0 of A
    ld1     {v0.4s}, [x1], #16       // A row 0
    
    fmul    v20.4s, v16.4s, v0.s[0]  // B_row0 * A[0][0]
    fmla    v20.4s, v17.4s, v0.s[1]  // += B_row1 * A[0][1]
    fmla    v20.4s, v18.4s, v0.s[2]  // += B_row2 * A[0][2]
    fmla    v20.4s, v19.4s, v0.s[3]  // += B_row3 * A[0][3]
    
    // Row 1 of A
    ld1     {v1.4s}, [x1], #16       // A row 1
    
    fmul    v21.4s, v16.4s, v1.s[0]
    fmla    v21.4s, v17.4s, v1.s[1]
    fmla    v21.4s, v18.4s, v1.s[2]
    fmla    v21.4s, v19.4s, v1.s[3]
    
    // Row 2 of A
    ld1     {v2.4s}, [x1], #16       // A row 2
    
    fmul    v22.4s, v16.4s, v2.s[0]
    fmla    v22.4s, v17.4s, v2.s[1]
    fmla    v22.4s, v18.4s, v2.s[2]
    fmla    v22.4s, v19.4s, v2.s[3]
    
    // Row 3 of A
    ld1     {v3.4s}, [x1], #16       // A row 3
    
    fmul    v23.4s, v16.4s, v3.s[0]
    fmla    v23.4s, v17.4s, v3.s[1]
    fmla    v23.4s, v18.4s, v3.s[2]
    fmla    v23.4s, v19.4s, v3.s[3]
    
    // Store result matrix C
    st1     {v20.4s}, [x0], #16      // C row 0
    st1     {v21.4s}, [x0], #16      // C row 1
    st1     {v22.4s}, [x0], #16      // C row 2
    st1     {v23.4s}, [x0], #16      // C row 3
    
    ret
