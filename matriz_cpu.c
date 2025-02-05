#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 1024  // Tamanho da matriz
#define NUM_TESTES 10  // Número de repetições para cálculo da média

void multiply_matrices(double A[N][N], double B[N][N], double C[N][N]) {
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
    static double A[N][N], B[N][N], C[N][N];
    srand(time(NULL));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = rand() % 100;

    double total_time = 0;
    
    for (int t = 0; t < NUM_TESTES; t++) {
        double start = omp_get_wtime();
        multiply_matrices(A, B, C);
        double end = omp_get_wtime();
        double exec_time = end - start;
        total_time += exec_time;
        printf("Teste %d: %f segundos\n", t + 1, exec_time);
    }

    printf("Tempo médio de execução na CPU: %f segundos\n", total_time / NUM_TESTES);
    return 0;
}
