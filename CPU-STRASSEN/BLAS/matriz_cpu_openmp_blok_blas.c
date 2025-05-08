#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>  // Inclui a interface C para BLAS

// Variáveis globais
int N;
double *A, *B, *C;  // Matrizes como arrays unidimensionais para BLAS
int num_threads;
int block_size;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Aloca matriz como array unidimensional para BLAS
double* allocate_matrix(int size) {
    return (double*)malloc(size * size * sizeof(double));
}

void free_matrix(double* matrix) {
    free(matrix);
}

void add_matrix(double* A, double* B, double* C, int size) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] + B[i];
    }
}

void sub_matrix(double* A, double* B, double* C, int size) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size * size; i++) {
        C[i] = A[i] - B[i];
    }
}

void matmul_blas(double* A, double* B, double* C, int size) {
    // Usando cblas_dgemm para multiplicação de matrizes
    // C = alpha * op(A) * op(B) + beta * C
    // Aqui: op(A) = A, op(B) = B, alpha = 1.0, beta = 0.0
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, A, size, B, size, 0.0, C, size);
}

void strassen(double* A, double* B, double* C, int size) {
    if (size <= block_size) {
        matmul_blas(A, B, C, size);
        return;
    }

    int newSize = size / 2;
    int offset = newSize * newSize;

    // Aloca submatrizes
    double *A11 = allocate_matrix(newSize);
    double *A12 = allocate_matrix(newSize);
    double *A21 = allocate_matrix(newSize);
    double *A22 = allocate_matrix(newSize);
    double *B11 = allocate_matrix(newSize);
    double *B12 = allocate_matrix(newSize);
    double *B21 = allocate_matrix(newSize);
    double *B22 = allocate_matrix(newSize);
    double *C11 = allocate_matrix(newSize);
    double *C12 = allocate_matrix(newSize);
    double *C21 = allocate_matrix(newSize);
    double *C22 = allocate_matrix(newSize);
    double *M1 = allocate_matrix(newSize);
    double *M2 = allocate_matrix(newSize);
    double *M3 = allocate_matrix(newSize);
    double *M4 = allocate_matrix(newSize);
    double *M5 = allocate_matrix(newSize);
    double *M6 = allocate_matrix(newSize);
    double *M7 = allocate_matrix(newSize);
    double *T1 = allocate_matrix(newSize);
    double *T2 = allocate_matrix(newSize);

    // Divisão das matrizes em blocos
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            int idx = i * newSize + j;
            int orig_idx = i * size + j;
            
            A11[idx] = A[orig_idx];
            A12[idx] = A[orig_idx + newSize];
            A21[idx] = A[orig_idx + newSize * size];
            A22[idx] = A[orig_idx + newSize * size + newSize];
            
            B11[idx] = B[orig_idx];
            B12[idx] = B[orig_idx + newSize];
            B21[idx] = B[orig_idx + newSize * size];
            B22[idx] = B[orig_idx + newSize * size + newSize];
        }
    }

    // Cálculo dos produtos M1-M7 usando BLAS
    // M1 = (A11 + A22)(B11 + B22)
    add_matrix(A11, A22, T1, newSize);
    add_matrix(B11, B22, T2, newSize);
    strassen(T1, T2, M1, newSize);
    
    // M2 = (A21 + A22)B11
    add_matrix(A21, A22, T1, newSize);
    strassen(T1, B11, M2, newSize);
    
    // M3 = A11(B12 - B22)
    sub_matrix(B12, B22, T1, newSize);
    strassen(A11, T1, M3, newSize);
    
    // M4 = A22(B21 - B11)
    sub_matrix(B21, B11, T1, newSize);
    strassen(A22, T1, M4, newSize);
    
    // M5 = (A11 + A12)B22
    add_matrix(A11, A12, T1, newSize);
    strassen(T1, B22, M5, newSize);
    
    // M6 = (A21 - A11)(B11 + B12)
    sub_matrix(A21, A11, T1, newSize);
    add_matrix(B11, B12, T2, newSize);
    strassen(T1, T2, M6, newSize);
    
    // M7 = (A12 - A22)(B21 + B22)
    sub_matrix(A12, A22, T1, newSize);
    add_matrix(B21, B22, T2, newSize);
    strassen(T1, T2, M7, newSize);

    // C11 = M1 + M4 - M5 + M7
    add_matrix(M1, M4, T1, newSize);
    sub_matrix(T1, M5, T2, newSize);
    add_matrix(T2, M7, C11, newSize);
    
    // C12 = M3 + M5
    add_matrix(M3, M5, C12, newSize);
    
    // C21 = M2 + M4
    add_matrix(M2, M4, C21, newSize);
    
    // C22 = M1 - M2 + M3 + M6
    sub_matrix(M1, M2, T1, newSize);
    add_matrix(T1, M3, T2, newSize);
    add_matrix(T2, M6, C22, newSize);

    // Combinação dos resultados na matriz C
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            int idx = i * newSize + j;
            int orig_idx = i * size + j;
            
            C[orig_idx] = C11[idx];
            C[orig_idx + newSize] = C12[idx];
            C[orig_idx + newSize * size] = C21[idx];
            C[orig_idx + newSize * size + newSize] = C22[idx];
        }
    }

    // Liberação de memória
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(C11); free(C12); free(C21); free(C22);
    free(M1); free(M2); free(M3); free(M4);
    free(M5); free(M6); free(M7); free(T1); free(T2);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <tamanho_matriz> <num_threads> <block_size>\n", argv[0]);
        printf("Exemplo: %s 1024 4 64\n");
        return 1;
    }

    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    block_size = atoi(argv[3]);

    if ((N & (N - 1)) != 0) {
        printf("Erro: o tamanho da matriz deve ser uma potência de 2 (ex: 256, 512, 1024...)\n");
        return 1;
    }

    if ((block_size & (block_size - 1)) != 0) {
        printf("Erro: o tamanho do bloco deve ser uma potência de 2 (ex: 32, 64, 128...)\n");
        return 1;
    }

    if (block_size > N) {
        printf("Erro: o tamanho do bloco não pode ser maior que o tamanho da matriz\n");
        return 1;
    }

    A = allocate_matrix(N);
    B = allocate_matrix(N);
    C = allocate_matrix(N);

    // Inicialização com valores aleatórios
    srand(1);
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
        C[i] = 0.0;
    }

    double start = get_time();
    strassen(A, B, C, N);
    double end = get_time();
    printf("Strassen com BLAS (Threads: %d, Block Size: %d): Tempo = %.4f s\n", 
           num_threads, block_size, end - start);

    free(A);
    free(B);
    free(C);

    return 0;
}