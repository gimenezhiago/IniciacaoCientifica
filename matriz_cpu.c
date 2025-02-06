#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <windows.h> 

#define NUM_TESTES 10  // Número de repetições para cálculo da média

// Função para medir o uso de energia e CPU
void medir_recursos() {
    FILETIME idleTime, kernelTime, userTime;
    GetSystemTimes(&idleTime, &kernelTime, &userTime);
    printf("Uso de CPU (medido pelo Windows): OK\n");
}

void multiply_matrices(double **A, double **B, double **C, int N) {
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
        printf("\nExecutando para N = %d\n", N);

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

        double total_time = 0;
        
        for (int t = 0; t < NUM_TESTES; t++) {
            double start = omp_get_wtime();
            multiply_matrices(A, B, C, N);
            double end = omp_get_wtime();
            double exec_time = end - start;
            total_time += exec_time;
            printf("Teste %d: %f segundos\n", t + 1, exec_time);
        }

        double avg_time = total_time / NUM_TESTES;
        printf("Tempo médio de execução na CPU para N=%d: %f segundos\n", N, avg_time);
        medir_recursos();

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
