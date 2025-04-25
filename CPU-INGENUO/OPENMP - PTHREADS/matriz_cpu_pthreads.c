#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

// Variáveis globais
int N;
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

// Função para multiplicação de matrizes com Pthreads
void *multiply_matrices_pthread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int i = data->start; i < data->end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
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
        pthread_create(&threads[i], NULL, multiply_matrices_pthread, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <tamanho_matriz> <num_threads>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);

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

    // Teste Pthreads
    start = get_time();
    run_pthread();
    end = get_time();
    printf("Pthreads (%d threads): Tempo = %.4f s\n", num_threads, end - start);
    
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
