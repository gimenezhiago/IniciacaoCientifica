#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <immintrin.h> // Para instruções AVX

// Variáveis globais
int N;
double **A, **B, **C;
int num_threads;
int block_size;

typedef struct {
    int start, end;
} ThreadData;

// Função para alinhar memória para AVX (alinhamento de 32 bytes)
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

double **allocate_matrix(int size) {
    double **matrix = (double **)aligned_malloc(size * sizeof(double *), 32);
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)aligned_malloc(size * sizeof(double), 32);
    }
    return matrix;
}

void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void add_matrix(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j += 4) {
            __m256d a = _mm256_load_pd(&A[i][j]);
            __m256d b = _mm256_load_pd(&B[i][j]);
            __m256d c = _mm256_add_pd(a, b);
            _mm256_store_pd(&C[i][j], c);
        }
    }
}

void sub_matrix(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j += 4) {
            __m256d a = _mm256_load_pd(&A[i][j]);
            __m256d b = _mm256_load_pd(&B[i][j]);
            __m256d c = _mm256_sub_pd(a, b);
            _mm256_store_pd(&C[i][j], c);
        }
    }
}

void matmul_blocked_avx(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            for (int k = 0; k < size; k += block_size) {
                // Multiplicação de blocos com AVX
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
                }
            }
        }
    }
}

void strassen(double **A, double **B, double **C, int size) {
    if (size <= block_size) {
        matmul_blocked_avx(A, B, C, size);
        return;
    }

    int newSize = size / 2;

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

    // Divisão das matrizes em blocos com AVX
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j += 4) {
            __m256d a = _mm256_load_pd(&A[i][j]);
            __m256d b = _mm256_load_pd(&A[i][j + newSize]);
            _mm256_store_pd(&A11[i][j], a);
            _mm256_store_pd(&A12[i][j], b);
            
            a = _mm256_load_pd(&A[i + newSize][j]);
            b = _mm256_load_pd(&A[i + newSize][j + newSize]);
            _mm256_store_pd(&A21[i][j], a);
            _mm256_store_pd(&A22[i][j], b);
            
            a = _mm256_load_pd(&B[i][j]);
            b = _mm256_load_pd(&B[i][j + newSize]);
            _mm256_store_pd(&B11[i][j], a);
            _mm256_store_pd(&B12[i][j], b);
            
            a = _mm256_load_pd(&B[i + newSize][j]);
            b = _mm256_load_pd(&B[i + newSize][j + newSize]);
            _mm256_store_pd(&B21[i][j], a);
            _mm256_store_pd(&B22[i][j], b);
        }
    }

    // Cálculo dos produtos M1-M7
    strassen(A11, B11, M1, newSize);   // M1 = A11 * B11
    strassen(A12, B21, M2, newSize);   // M2 = A12 * B21
    strassen(A11, B12, M3, newSize);   // M3 = A11 * B12
    strassen(A12, B22, M4, newSize);   // M4 = A12 * B22
    strassen(A21, B11, M5, newSize);   // M5 = A21 * B11
    strassen(A22, B21, M6, newSize);   // M6 = A22 * B21
    strassen(A21, B12, M7, newSize);   // M7 = A21 * B12

    // Combinação dos resultados com AVX
    add_matrix(M1, M2, C11, newSize);
    add_matrix(M3, M4, C12, newSize);
    add_matrix(M5, M6, C21, newSize);
    add_matrix(M3, M7, C22, newSize);

    // Combinação dos resultados na matriz C com AVX
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j += 4) {
            __m256d c11 = _mm256_load_pd(&C11[i][j]);
            __m256d c12 = _mm256_load_pd(&C12[i][j]);
            __m256d c21 = _mm256_load_pd(&C21[i][j]);
            __m256d c22 = _mm256_load_pd(&C22[i][j]);
            
            _mm256_store_pd(&C[i][j], c11);
            _mm256_store_pd(&C[i][j + newSize], c12);
            _mm256_store_pd(&C[i + newSize][j], c21);
            _mm256_store_pd(&C[i + newSize][j + newSize], c22);
        }
    }

    // Liberação de memória
    free_matrix(A11, newSize); free_matrix(A12, newSize); free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize); free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize); free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(M1, newSize); free_matrix(M2, newSize); free_matrix(M3, newSize); free_matrix(M4, newSize);
    free_matrix(M5, newSize); free_matrix(M6, newSize); free_matrix(M7, newSize); free_matrix(T1, newSize); free_matrix(T2, newSize);
}

// Thread executa multiplicação de parte de A com B usando AVX
void *multiply_strassen_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < N; j += 4) {
            __m256d c = _mm256_setzero_pd();
            for (int k = 0; k < N; k++) {
                __m256d a = _mm256_broadcast_sd(&A[i][k]);
                __m256d b = _mm256_load_pd(&B[k][j]);
                c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
            }
            _mm256_store_pd(&C[i][j], c);
        }
    }
    return NULL;
}

void run_pthread() {
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk = N / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk;
        thread_data[i].end = (i == num_threads - 1) ? N : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, multiply_strassen_thread, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
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
        printf("Erro: o tamanho da matriz deve ser potência de 2 (ex: 256, 512...)\n");
        return 1;
    }

    if ((block_size & (block_size - 1)) != 0) {
        printf("Erro: o tamanho do bloco deve ser potência de 2 (ex: 32, 64...)\n");
        return 1;
    }

    if (block_size > N) {
        printf("Erro: o tamanho do bloco não pode ser maior que o tamanho da matriz\n");
        return 1;
    }

    A = allocate_matrix(N);
    B = allocate_matrix(N);
    C = allocate_matrix(N);

    // Inicialização com valores aleatórios usando AVX
    srand(1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            __m256d r = _mm256_set_pd(rand() % 100, rand() % 100, rand() % 100, rand() % 100);
            _mm256_store_pd(&A[i][j], r);
            r = _mm256_set_pd(rand() % 100, rand() % 100, rand() % 100, rand() % 100);
            _mm256_store_pd(&B[i][j], r);
        }
    }

    double start = get_time();
    run_pthread();
    double end = get_time();

    printf("Strassen com Pthreads e AVX (Threads: %d, Block Size: %d): Tempo = %.4f s\n", 
           num_threads, block_size, end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}