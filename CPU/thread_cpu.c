//Quants threads a CPU consegue executar

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <windows.h> 

#define NUM_TESTES 10  // Número de repetições para média

// Função de multiplicação de matrizes SEM OpenMP
void multiply_matrices_seq(double **A, double **B, double **C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Função de multiplicação de matrizes COM OpenMP
void multiply_matrices_omp(double **A, double **B, double **C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int tamanhos[] = {512, 1024, 2048}; // Diferentes tamanhos de entrada
    int num_tamanhos = sizeof(tamanhos) / sizeof(tamanhos[0]);

    for (int idx = 0; idx < num_tamanhos; idx++) {
        int N = tamanhos[idx];

        // Alocação dinâmica das matrizes
        double **A = (double **)malloc(N * sizeof(double *));
        double **B = (double **)malloc(N * sizeof(double *));
        double **C = (double **)malloc(N * sizeof(double *));
        for (int i = 0; i < N; i++) {
            A[i] = (double *)malloc(N * sizeof(double));
            B[i] = (double *)malloc(N * sizeof(double));
            C[i] = (double *)malloc(N * sizeof(double));
        }

        srand(time(NULL));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = B[i][j] = rand() % 100;

        double total_time_seq = 0, total_time_omp = 0;

        // Executando SEM OpenMP
        for (int t = 0; t < NUM_TESTES; t++) {
            double start = omp_get_wtime();
            multiply_matrices_seq(A, B, C, N);
            double end = omp_get_wtime();
            double exec_time = end - start;
            total_time_seq += exec_time;
        }

        // Obtendo o número de threads OpenMP
        int num_threads = 1;
        #pragma omp parallel
        {
            num_threads = omp_get_num_threads();
        }
        printf("Número de threads utilizadas: %d\n", num_threads);

        // Executando COM OpenMP
        for (int t = 0; t < NUM_TESTES; t++) {
            double start = omp_get_wtime();
            multiply_matrices_omp(A, B, C, N);
            double end = omp_get_wtime();
            double exec_time = end - start;
            total_time_omp += exec_time;
        }

        // Liberação da memória
        for (int i = 0; i < N; i++) {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }
        free(A);
        free(B);
        free(C);
    }

    return 0;
}
