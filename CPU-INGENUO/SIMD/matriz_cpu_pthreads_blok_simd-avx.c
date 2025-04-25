#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <immintrin.h> // Inclui as intrínsecas do AVX

// Variáveis globais
int N, block_size;
double **A, **B, **C;
int num_threads;

// Estrutura de dados para Pthreads
typedef struct {
    int start, end;
} ThreadData;

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Função para multiplicação de matrizes com Pthreads, blocking e AVX
void *multiply_matrices_pthread_avx(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < i + block_size && ii < data->end; ii++) {
                    for (int jj = j; jj < j + block_size && jj < N; jj++) {
                        __m256d c = _mm256_setzero_pd(); // Inicializa o acumulador AVX com zeros
                        for (int kk = k; kk < k + block_size && kk < N; kk += 4) {
                            // Carrega 4 elementos de A e B usando AVX
                            __m256d a = _mm256_loadu_pd(&A[ii][kk]);
                            __m256d b = _mm256_loadu_pd(&B[kk][jj]);
                            // Multiplica e acumula usando AVX
                            c = _mm256_add_pd(c, _mm256_mul_pd(a, b));
                        }
                        // Soma os elementos do vetor AVX e armazena em C[ii][jj]
                        double temp[4];
                        _mm256_storeu_pd(temp, c);
                        C[ii][jj] += temp[0] + temp[1] + temp[2] + temp[3];
                    }
                }
            }
        }
    }
    return NULL;
}

void run_pthread_avx() {
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk = N / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk;
        thread_data[i].end = (i == num_threads - 1) ? N : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, multiply_matrices_pthread_avx, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <tamanho_matriz> <num_threads> <blocking_size>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    block_size = atoi(argv[3]);

    // Alocação dinâmica das matrizes
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    // Inicialização das matrizes
    srand(1);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = rand() % 100;

    double start, end;

    // Teste Pthreads com AVX
    start = get_time();
    run_pthread_avx();
    end = get_time();
    printf("Pthreads + AVX (%d threads, bloco %d): Tempo = %.4f s\n", num_threads, block_size, end - start);
    
    // Liberação de memória
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}