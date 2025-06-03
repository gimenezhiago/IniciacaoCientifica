#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <cblas.h>  // Interface C do OpenBLAS

// Variáveis globais
int N, block_size;
double **A, **B, **C;
int num_threads;

// Estrutura para argumentos das threads
typedef struct {
    double *A;
    double *B;
    double *C;
    int size;
    int start_row;
    int end_row;
} MatrixArgs;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Aloca matriz como um único bloco contíguo para melhor desempenho com BLAS
double **allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    double *data = (double *)malloc(size * size * sizeof(double));
    for (int i = 0; i < size; i++) {
        matrix[i] = &data[i * size];
    }
    return matrix;
}

// Versão alternativa para matrizes separadas (usada nas submatrizes do Strassen)
double **allocate_submatrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)calloc(size, sizeof(double));
    }
    return matrix;
}

void free_matrix(double **matrix, int size) {
    if (size == N) {
        // Matriz principal tem dados contíguos
        free(matrix[0]);
    } else {
        // Submatrizes alocadas separadamente
        for (int i = 0; i < size; i++)
            free(matrix[i]);
    }
    free(matrix);
}

// Multiplicação usando OpenBLAS (dgemm)
void blas_mult(double *A, double *B, double *C, int size) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, A, size, B, size, 0.0, C, size);
}

// Função executada pelas threads (usando OpenBLAS)
void* threaded_mult_blocking_blas(void* arg) {
    MatrixArgs *args = (MatrixArgs *)arg;
    
    // Calcula offsets para submatrizes
    int offset_A = args->start_row * args->size;
    int offset_C = args->start_row * args->size;
    int rows = args->end_row - args->start_row;
    
    // Zera a região de C que será calculada
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < args->size; j++) {
            args->C[offset_C + i * args->size + j] = 0.0;
        }
    }
    
    // Usa OpenBLAS para multiplicar
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, args->size, args->size, 
                1.0, &args->A[offset_A], args->size,
                args->B, args->size,
                1.0, &args->C[offset_C], args->size);
    
    pthread_exit(NULL);
}

void strassen(double **A, double **B, double **C, int size) {
    if (size <= block_size) {
        pthread_t threads[num_threads];
        MatrixArgs args[num_threads];
        int rows_per_thread = size / num_threads;

        // Configura argumentos para threads
        for (int t = 0; t < num_threads; t++) {
            args[t].A = A[0];  // Acesso ao array contíguo
            args[t].B = B[0];
            args[t].C = C[0];
            args[t].size = size;
            args[t].start_row = t * rows_per_thread;
            args[t].end_row = (t == num_threads - 1) ? size : (t + 1) * rows_per_thread;
            pthread_create(&threads[t], NULL, threaded_mult_blocking_blas, &args[t]);
        }

        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        return;
    }

    int newSize = size / 2;

    // Aloca submatrizes
    double **A11 = allocate_submatrix(newSize);
    double **A12 = allocate_submatrix(newSize);
    double **A21 = allocate_submatrix(newSize);
    double **A22 = allocate_submatrix(newSize);
    double **B11 = allocate_submatrix(newSize);
    double **B12 = allocate_submatrix(newSize);
    double **B21 = allocate_submatrix(newSize);
    double **B22 = allocate_submatrix(newSize);
    double **C11 = allocate_submatrix(newSize);
    double **C12 = allocate_submatrix(newSize);
    double **C21 = allocate_submatrix(newSize);
    double **C22 = allocate_submatrix(newSize);
    double **M1 = allocate_submatrix(newSize);
    double **M2 = allocate_submatrix(newSize);
    double **M3 = allocate_submatrix(newSize);
    double **M4 = allocate_submatrix(newSize);
    double **M5 = allocate_submatrix(newSize);
    double **M6 = allocate_submatrix(newSize);
    double **M7 = allocate_submatrix(newSize);
    double **T1 = allocate_submatrix(newSize);
    double **T2 = allocate_submatrix(newSize);

    // Preenche submatrizes
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // Recursão (Strassen)
    strassen(A11, B11, M1, newSize);
    strassen(A21, B11, M2, newSize);
    strassen(A11, B12, M3, newSize);
    strassen(A22, B21, M4, newSize);
    strassen(A11, B22, M5, newSize);
    strassen(A21, B12, M6, newSize);
    strassen(A12, B21, M7, newSize);

    // Combina resultados usando OpenBLAS para operações matriciais
    // T1 = M1 + M4
    for (int i = 0; i < newSize; i++) {
        cblas_daxpy(newSize, 1.0, M1[i], 1, T1[i], 1);
        cblas_daxpy(newSize, 1.0, M4[i], 1, T1[i], 1);
    }
    // T2 = T1 - M5
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, T1[i], 1, T2[i], 1);
        cblas_daxpy(newSize, -1.0, M5[i], 1, T2[i], 1);
    }
    // C11 = T2 + M7
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, T2[i], 1, C11[i], 1);
        cblas_daxpy(newSize, 1.0, M7[i], 1, C11[i], 1);
    }

    // C12 = M3 + M5
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, M3[i], 1, C12[i], 1);
        cblas_daxpy(newSize, 1.0, M5[i], 1, C12[i], 1);
    }

    // C21 = M2 + M4
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, M2[i], 1, C21[i], 1);
        cblas_daxpy(newSize, 1.0, M4[i], 1, C21[i], 1);
    }

    // T1 = M1 - M2
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, M1[i], 1, T1[i], 1);
        cblas_daxpy(newSize, -1.0, M2[i], 1, T1[i], 1);
    }
    // T2 = T1 + M3
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, T1[i], 1, T2[i], 1);
        cblas_daxpy(newSize, 1.0, M3[i], 1, T2[i], 1);
    }
    // C22 = T2 + M6
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, T2[i], 1, C22[i], 1);
        cblas_daxpy(newSize, 1.0, M6[i], 1, C22[i], 1);
    }

    // Copia para C
    for (int i = 0; i < newSize; i++) {
        cblas_dcopy(newSize, C11[i], 1, C[i], 1);
        cblas_dcopy(newSize, C12[i], 1, &C[i][newSize], 1);
        cblas_dcopy(newSize, C21[i], 1, C[i + newSize], 1);
        cblas_dcopy(newSize, C22[i], 1, &C[i + newSize][newSize], 1);
    }

    // Libera memória
    free_matrix(A11, newSize); free_matrix(A12, newSize); free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize); free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize); free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(M1, newSize);  free_matrix(M2, newSize);  free_matrix(M3, newSize);  free_matrix(M4, newSize);
    free_matrix(M5, newSize);  free_matrix(M6, newSize);  free_matrix(M7, newSize);  free_matrix(T1, newSize);  free_matrix(T2, newSize);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <tamanho_matriz> <block_size> <num_threads>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    block_size = atoi(argv[2]);
    num_threads = atoi(argv[3]);

    if ((N & (N - 1)) != 0 || (block_size & (block_size - 1)) != 0) {
        printf("Erro: tamanho_matriz e block_size devem ser potências de 2 (ex: 64, 128, 256...)\n");
        return 1;
    }

    // Configura número de threads para OpenBLAS
    openblas_set_num_threads(1);  // Nós controlamos o paralelismo com Pthreads

    A = allocate_matrix(N);
    B = allocate_matrix(N);
    C = allocate_matrix(N);

    // Inicializa matrizes com valores aleatórios
    srand(1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    double start = get_time();
    strassen(A, B, C, N);
    double end = get_time();
    printf("Strassen com Pthreads + OpenBLAS (block_size=%d, %d threads): Tempo = %.4f s\n", 
           block_size, num_threads, end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}