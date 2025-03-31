const char* sgemm_desc = "Simple blocked sgemm with folded kernels.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define SMALL_BLOCK_SIZE 16
#define SMALL_BLOCK_M_SIZE 32
#define SMALL_BLOCK_N_SIZE 8
#define min(a,b) (((a)<(b))?(a):(b))

// 朴素矩阵乘法，处理小块和边界情况
static void do_block_opt(int lda, int M, int N, int K, float * restrict A, float * restrict B, float * restrict C)
{
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) {
            register float b = B[k + j * lda];
            for (int i = 0; i < M; ++i)
                C[i + j * lda] += A[i + k * lda] * b;
        }
}

// 64×1矩阵乘法的AVX-512实现
static void do_block_avx_64k1(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[4]; 

    for (int block = 0; block < 4; block++) {
        c[block] = _mm512_load_ps(&C[block * 16]);
    }

    __m512 a[4];
    __m512 b;

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 4; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        b = _mm512_set1_ps(B[k + ldb * 0]);

        for (int block = 0; block < 4; block++) {
            c[block] = _mm512_fmadd_ps(a[block], b, c[block]);
        }
    }

    for (int block = 0; block < 4; block++) {
        _mm512_store_ps(&C[block * 16], c[block]);
    }
}

// 16×16矩阵乘法的AVX-512实现
static void do_block_avx_16k16(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[16];

    for (int j = 0; j < 16; j++) {
        c[j] = _mm512_load_ps(&C[ldc * j]);
    }

    __m512 a;
    __m512 b;

    for (int k = 0; k < K; k++) {
        a = _mm512_load_ps(&A[lda * k]);

        for (int j = 0; j < 16; j++) {
            b = _mm512_set1_ps(B[k + ldb * j]);
            c[j] = _mm512_fmadd_ps(a, b, c[j]);
        }
    }

    for (int j = 0; j < 16; j++) {
        _mm512_store_ps(&C[ldc * j], c[j]);
    }
}

// 32×8矩阵乘法的AVX-512实现
static void do_block_avx_32k8(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[2][8];

    for (int block = 0; block < 2; block++) {
        for (int j = 0; j < 8; j++) {
            c[block][j] = _mm512_load_ps(&C[ldc * j + block * 16]);
        }
    }

    __m512 a[2];
    __m512 b[8];

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 2; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        for (int j = 0; j < 8; j++) {
            b[j] = _mm512_set1_ps(B[k + ldb * j]);
        }

        for (int block = 0; block < 2; block++) {
            for (int j = 0; j < 8; j++) {
                c[block][j] = _mm512_fmadd_ps(a[block], b[j], c[block][j]);
            }
        }
    }

    for (int block = 0; block < 2; block++) {
        for (int j = 0; j < 8; j++) {
            _mm512_store_ps(&C[ldc * j + block * 16], c[block][j]);
        }
    }
}

// 48×8矩阵乘法的AVX-512实现
static void do_block_avx_48k8(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[3][8];

    for (int block = 0; block < 3; block++) {
        for (int j = 0; j < 8; j++) {
            c[block][j] = _mm512_load_ps(&C[ldc * j + block * 16]);
        }
    }

    __m512 a[3];
    __m512 b[8];

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 3; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        for (int j = 0; j < 8; j++) {
            b[j] = _mm512_set1_ps(B[k + ldb * j]);
        }

        for (int block = 0; block < 3; block++) {
            for (int j = 0; j < 8; j++) {
                c[block][j] = _mm512_fmadd_ps(a[block], b[j], c[block][j]);
            }
        }
    }

    for (int block = 0; block < 3; block++) {
        for (int j = 0; j < 8; j++) {
            _mm512_store_ps(&C[ldc * j + block * 16], c[block][j]);
        }
    }
}

// 64×4矩阵乘法的AVX-512实现
static void do_block_avx_64k4(int lda, int ldb, int ldc, int K, float * restrict A, float * restrict B, float * restrict C)
{
    __m512 c[4][4];

    for (int block = 0; block < 4; block++) {
        for (int j = 0; j < 4; j++) {
            c[block][j] = _mm512_load_ps(&C[ldc * j + block * 16]);
        }
    }

    __m512 a[4];
    __m512 b[4];

    for (int k = 0; k < K; k++) {
        for (int block = 0; block < 4; block++) {
            a[block] = _mm512_load_ps(&A[lda * k + block * 16]);
        }

        for (int j = 0; j < 4; j++) {
            b[j] = _mm512_set1_ps(B[k + ldb * j]);
        }

        for (int block = 0; block < 4; block++) {
            for (int j = 0; j < 4; j++) {
                c[block][j] = _mm512_fmadd_ps(a[block], b[j], c[block][j]);
            }
        }
    }

    for (int block = 0; block < 4; block++) {
        for (int j = 0; j < 4; j++) {
            _mm512_store_ps(&C[ldc * j + block * 16], c[block][j]);
        }
    }
}

// 根据矩阵大小选择最佳内核处理大块
static void do_block_large(int lda, int M, int N, int K, float* restrict A, float* restrict B, float* restrict C)
{
    if((M % SMALL_BLOCK_M_SIZE == 0) && (N % SMALL_BLOCK_N_SIZE == 0))
    {
        for (int j = 0; j < N; j += SMALL_BLOCK_N_SIZE)
            for (int i = 0; i < M; i += SMALL_BLOCK_M_SIZE)
                do_block_avx_32k8(lda, lda, lda, K, A + i, B + j * lda, C + i + j * lda);
        return;
    }
    
    if (N == 1) {
        if (M == 64) {
            do_block_avx_64k1(lda, lda, lda, K, A, B, C);
            return;
        }
    }

    float* restrict AA = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_SIZE * K, 64);
    float* restrict BB = (float*)_mm_malloc(sizeof(float) * K * SMALL_BLOCK_SIZE, 64);
    float* restrict CC = (float*)_mm_malloc(sizeof(float) * SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE, 64);

    int M_Left = M % SMALL_BLOCK_SIZE;
    int N_Left = N % SMALL_BLOCK_SIZE;

    if (M_Left > 0) {
        for(int j = 0; j < K; j++) {
            __m512 Avec = _mm512_load_ps(&A[M - M_Left + j * lda]);
            _mm512_store_ps(&AA[j * SMALL_BLOCK_SIZE], Avec);
        }
    }

    if (N_Left > 0) {
        for(int j = 0; j < SMALL_BLOCK_SIZE; j++) {
            int i;
            for(i = 0; i < K - 15; i += 16) {
                __m512 Bvec = _mm512_load_ps(&B[i + (N - N_Left + j) * lda]);
                _mm512_store_ps(&BB[i + j * K], Bvec);
            }
            for(; i < K; i++)
                BB[i + j * K] = B[i + (N - N_Left + j) * lda];
        }
    }

    for (int j = 0; j < N; j += SMALL_BLOCK_SIZE) {
        int N_part = min(SMALL_BLOCK_SIZE, N - j);
        for (int i = 0; i < M; i += SMALL_BLOCK_SIZE) {
            int M_part = min(SMALL_BLOCK_SIZE, M - i);

            if (M_part == SMALL_BLOCK_SIZE && N_part == SMALL_BLOCK_SIZE) {
                do_block_avx_16k16(lda, lda, lda, K, A + i, B + j * lda, C + i + j * lda);
            }
            else if(N_part == SMALL_BLOCK_SIZE) {
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
                    
                do_block_avx_16k16(SMALL_BLOCK_SIZE, lda, SMALL_BLOCK_SIZE, K, AA, B + j * lda, CC);
                
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
            }
            else if(M_part == SMALL_BLOCK_SIZE) {
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
                    
                do_block_avx_16k16(lda, K, SMALL_BLOCK_SIZE, K, A + i, BB, CC);
                
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
            }
            else {
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii+i) + (jj+j) * lda];
                    
                do_block_avx_16k16(SMALL_BLOCK_SIZE, K, SMALL_BLOCK_SIZE, K, AA, BB, CC);
                
                for(int jj = 0; jj < N_part; jj++)
                    for(int ii = 0; ii < M_part; ii++)
                        C[(ii+i) + (jj+j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
            }  
        }
    }

    _mm_free(AA);
    _mm_free(BB);
    _mm_free(CC);
}

// 矩阵乘法主函数，处理整个矩阵
void square_sgemm(int lda, float* restrict A, float* restrict B, float* restrict C)
{
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
        int N = min(BLOCK_SIZE, lda-j);
        for (int i = 0; i < lda; i += BLOCK_SIZE) {
            int M = min(BLOCK_SIZE, lda-i);
            do_block_large(lda, M, N, lda, A + i, B + j*lda, C + i + j*lda);
        }
    }
}