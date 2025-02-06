#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NUM_TESTES 5  // Número de repetições para maior precisão

// Função para multiplicação de matrizes em paralelo
void multiply_matrices(double **A, double **B, double **C, int N, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Função para medir o overhead de criação de threads
void medir_overhead_matrizes(int N, int num_threads) {
    double overhead_time = 0, exec_time = 0;

    // Alocar memória para as matrizes
    double **A = (double **)malloc(N * sizeof(double *));
    double **B = (double **)malloc(N * sizeof(double *));
    double **C = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    // Inicializar matrizes com valores aleatórios
    srand(time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = rand() % 100;

    for (int t = 0; t < NUM_TESTES; t++) {
        // Medir tempo de criação das threads sem cálculo significativo
        double start_overhead = omp_get_wtime();
        #pragma omp parallel num_threads(num_threads)
        {
            // Apenas para criar/destruir threads
        }
        double end_overhead = omp_get_wtime();
        overhead_time += (end_overhead - start_overhead);

        // Medir tempo real da multiplicação de matrizes
        double start_exec = omp_get_wtime();
        multiply_matrices(A, B, C, N, num_threads);
        double end_exec = omp_get_wtime();
        exec_time += (end_exec - start_exec);
    }

    // Média dos tempos
    overhead_time /= NUM_TESTES;
    exec_time /= NUM_TESTES;

    // Exibir resultados
    printf("\nTamanho da matriz: %d x %d | Threads: %d\n", N, N, num_threads);
    printf("Tempo médio para criar/destruir threads: %f segundos\n", overhead_time);
    printf("Tempo médio para multiplicação de matrizes: %f segundos\n", exec_time);
    printf("Overhead relativo (overhead / execução real): %f%%\n", (overhead_time / exec_time) * 100);

    // Liberar memória
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
}

int main() {
    int tamanhos[] = {512, 1024}; // Testar com diferentes tamanhos de matrizes
    int threads[] = {1, 2, 4, 8}; // Testar com diferentes quantidades de threads

    int num_tamanhos = sizeof(tamanhos) / sizeof(tamanhos[0]);
    int num_threads = sizeof(threads) / sizeof(threads[0]);

    for (int i = 0; i < num_tamanhos; i++) {
        for (int j = 0; j < num_threads; j++) {
            medir_overhead_matrizes(tamanhos[i], threads[j]);
        }
    }

    return 0;
}
