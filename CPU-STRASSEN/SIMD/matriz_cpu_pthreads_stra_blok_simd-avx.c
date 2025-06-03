#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <immintrin.h>  // Para instruções AVX

// Variáveis globais
int N, block_size;
double **A, **B, **C;
int num_threads;

// Estrutura para argumentos das threads
typedef struct {
    double **A;
    double **B;
    double **C;
    int size;
    int start_row;
    int end_row;
} MatrixArgs;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

double **allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)aligned_alloc(32, size * sizeof(double));  // Alinhamento para AVX
    }
    return matrix;
}

void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++)
        free(matrix[i]);
    free(matrix);
}

// Função de adição otimizada com AVX
void add_matrix_avx(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++) {
        int j;
        for (j = 0; j <= size - 4; j += 4) {
            __m256d a = _mm256_load_pd(&A[i][j]);
            __m256d b = _mm256_load_pd(&B[i][j]);
            __m256d c = _mm256_add_pd(a, b);
            _mm256_store_pd(&C[i][j], c);
        }
        // Processa os elementos restantes
        for (; j < size; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Função de subtração otimizada com AVX
void sub_matrix_avx(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++) {
        int j;
        for (j = 0; j <= size - 4; j += 4) {
            __m256d a = _mm256_load_pd(&A[i][j]);
            __m256d b = _mm256_load_pd(&B[i][j]);
            __m256d c = _mm256_sub_pd(a, b);
            _mm256_store_pd(&C[i][j], c);
        }
        // Processa os elementos restantes
        for (; j < size; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Multiplicação com blocking e AVX
void conventional_mult_blocking_avx(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            for (int k = 0; k < size; k += block_size) {
                // Processa blocos com AVX
                for (int ii = i; ii < i + block_size && ii < size; ii++) {
                    for (int jj = j; jj < j + block_size && jj < size; jj += 4) {
                        __m256d c = _mm256_load_pd(&C[ii][jj]);
                        for (int kk = k; kk < k + block_size && kk < size; kk++) {
                            __m256d a = _mm256_broadcast_sd(&A[ii][kk]);
                            __m256d b = _mm256_load_pd(&B[kk][jj]);
                            c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
                        }
                        _mm256_store_pd(&C[ii][jj], c);
                    }
                    // Processa colunas restantes
                    for (int jj = j + ((j + block_size) / 4) * 4; jj < j + block_size && jj < size; jj++) {
                        double sum = C[ii][jj];
                        for (int kk = k; kk < k + block_size && kk < size; kk++) {
                            sum += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] = sum;
                    }
                }
            }
        }
    }
}

// Função executada pelas threads (com AVX)
void* threaded_mult_blocking_avx(void* arg) {
    MatrixArgs *args = (MatrixArgs *)arg;
    for (int i = args->start_row; i < args->end_row; i += block_size) {
        for (int j = 0; j < args->size; j += block_size) {
            for (int k = 0; k < args->size; k += block_size) {
                // Processa blocos com AVX
                for (int ii = i; ii < i + block_size && ii < args->end_row && ii < args->size; ii++) {
                    for (int jj = j; jj < j + block_size && jj < args->size; jj += 4) {
                        __m256d c = _mm256_load_pd(&args->C[ii][jj]);
                        for (int kk = k; kk < k + block_size && kk < args->size; kk++) {
                            __m256d a = _mm256_broadcast_sd(&args->A[ii][kk]);
                            __m256d b = _mm256_load_pd(&args->B[kk][jj]);
                            c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
                        }
                        _mm256_store_pd(&args->C[ii][jj], c);
                    }
                    // Processa colunas restantes
                    for (int jj = j + ((j + block_size) / 4) * 4; jj < j + block_size && jj < args->size; jj++) {
                        double sum = args->C[ii][jj];
                        for (int kk = k; kk < k + block_size && kk < args->size; kk++) {
                            sum += args->A[ii][kk] * args->B[kk][jj];
                        }
                        args->C[ii][jj] = sum;
                    }
                }
            }
        }
    }
    pthread_exit(NULL);
}

void strassen(double **A, double **B, double **C, int size) {
    if (size <= block_size) {
        pthread_t threads[num_threads];
        MatrixArgs args[num_threads];
        int rows_per_thread = size / num_threads;

        // Inicializa C com zeros
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                C[i][j] = 0.0;
            }
        }

        // Divide o trabalho entre threads
        for (int t = 0; t < num_threads; t++) {
            args[t].A = A;
            args[t].B = B;
            args[t].C = C;
            args[t].size = size;
            args[t].start_row = t * rows_per_thread;
            args[t].end_row = (t == num_threads - 1) ? size : (t + 1) * rows_per_thread;
            pthread_create(&threads[t], NULL, threaded_mult_blocking_avx, &args[t]);
        }

        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }

        return;
    }

    int newSize = size / 2;

    // Aloca submatrizes
    double **A11 = allocate_matrix(newSize);
    double **A12 = allocate_matrix(newSize);
    double **A21 = allocate_matrix(newSize);
    double **A22 = allocate_matrix(newSize);
    double **B11 = allocate_matrix(newSize);
    double **B12 = allocate_matrix(newSize);
    double **B21 = allocate_matrix(newSize);
    double **B22 = allocate_matrix(newSize);
    double **C11 = allocate_matrix(newSize);
    double **C12 = allocate_matrix(newSize);
    double **C21 = allocate_matrix(newSize);
    double **C22 = allocate_matrix(newSize);
    double **M1 = allocate_matrix(newSize);
    double **M2 = allocate_matrix(newSize);
    double **M3 = allocate_matrix(newSize);
    double **M4 = allocate_matrix(newSize);
    double **M5 = allocate_matrix(newSize);
    double **M6 = allocate_matrix(newSize);
    double **M7 = allocate_matrix(newSize);
    double **T1 = allocate_matrix(newSize);
    double **T2 = allocate_matrix(newSize);

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

    // Combina resultados com AVX
    add_matrix_avx(M1, M4, T1, newSize);
    sub_matrix_avx(T1, M5, T2, newSize);
    add_matrix_avx(T2, M7, C11, newSize);

    add_matrix_avx(M3, M5, C12, newSize);
    add_matrix_avx(M2, M4, C21, newSize);

    sub_matrix_avx(M1, M2, T1, newSize);
    add_matrix_avx(T1, M3, T2, newSize);
    add_matrix_avx(T2, M6, C22, newSize);

    // Copia para C
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
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
    printf("Strassen com Pthreads + Blocking + AVX (block_size=%d, %d threads): Tempo = %.4f s\n", 
           block_size, num_threads, end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}