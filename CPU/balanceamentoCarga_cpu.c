#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NUM_TESTES 5  // Número de repetições para maior precisão

// Função de multiplicação de matrizes com medição de tempo por thread
void multiply_matrices(double **A, double **B, double **C, int N, int num_threads, char *schedule_type) {
    double start_times[num_threads], end_times[num_threads]; // Guardar tempos individuais das threads

    // Inicializar tempos como 0
    for (int i = 0; i < num_threads; i++) {
        start_times[i] = 0;
        end_times[i] = 0;
    }

    // Medir tempo de execução por thread
    double start_exec = omp_get_wtime();
    
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        start_times[thread_id] = omp_get_wtime(); // Marca o início da thread

        // Multiplicação de matrizes com diferentes estratégias de schedule
        #pragma omp for schedule(dynamic) // Troque para static, dynamic ou guided
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        end_times[thread_id] = omp_get_wtime(); // Marca o fim da thread
    }

    double end_exec = omp_get_wtime();
    double exec_time = end_exec - start_exec;

    // Exibir resultados
    printf("\nTamanho da matriz: %d x %d | Threads: %d | Schedule: %s\n", N, N, num_threads, schedule_type);
    printf("Tempo total de execução: %f segundos\n", exec_time);
    
    // Exibir tempos individuais de cada thread
    for (int i = 0; i < num_threads; i++) {
        if (start_times[i] != 0) { // Apenas mostrar threads que foram usadas
            printf("Thread %d - Tempo de trabalho: %f segundos\n", i, end_times[i] - start_times[i]);
        }
    }
}

int main() {
    int tamanhos[] = {512}; // Testar apenas com 512x512 para facilitar a análise
    int threads[] = {2, 4, 8}; // Testar com diferentes quantidades de threads
    char *schedules[] = {"static", "dynamic", "guided"}; // Diferentes estratégias de balanceamento

    int num_tamanhos = sizeof(tamanhos) / sizeof(tamanhos[0]);
    int num_threads = sizeof(threads) / sizeof(threads[0]);
    int num_schedules = sizeof(schedules) / sizeof(schedules[0]);

    for (int i = 0; i < num_tamanhos; i++) {
        int N = tamanhos[i];

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

        for (int j = 0; j < num_threads; j++) {
            for (int k = 0; k < num_schedules; k++) {
                multiply_matrices(A, B, C, N, threads[j], schedules[k]);
            }
        }

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

    return 0;
}
