#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <windows.h>
#include <psapi.h>
#include <locale.h>

#define NUM_TESTES 10

// Função para obter uso de memória em MB
double get_memory_usage() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0.0;
}

// Função para obter uso de CPU em %
double get_cpu_usage() {
    static ULARGE_INTEGER last_idle_time = {0};
    static ULARGE_INTEGER last_kernel_time = {0};
    static ULARGE_INTEGER last_user_time = {0};
    
    FILETIME idleTime, kernelTime, userTime;
    
    if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        ULARGE_INTEGER k, u, i;
        k.LowPart = kernelTime.dwLowDateTime;
        k.HighPart = kernelTime.dwHighDateTime;
        u.LowPart = userTime.dwLowDateTime;
        u.HighPart = userTime.dwHighDateTime;
        i.LowPart = idleTime.dwLowDateTime;
        i.HighPart = idleTime.dwHighDateTime;

        // Calcular a diferença desde a última medição
        double total_diff = (double)((k.QuadPart - last_kernel_time.QuadPart) + 
                                     (u.QuadPart - last_user_time.QuadPart));
        double idle_diff = (double)(i.QuadPart - last_idle_time.QuadPart);
        
        // Atualizar o último tempo
        last_kernel_time = k;
        last_user_time = u;
        last_idle_time = i;
        
        // Calcular e retornar a porcentagem de CPU
        return 100.0 * (total_diff - idle_diff) / total_diff;
    }
    
    return 0.0;
}

// Função para obter o TSC (Time-Stamp Counter)
unsigned long long get_tsc() {
    unsigned int low, high;
    __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high));
    return ((unsigned long long)high << 32) | low;
}

// Função para calcular a frequência da CPU
double calculate_cpu_frequency() {
    unsigned long long start, end;
    LARGE_INTEGER freq, start_time, end_time;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start_time);
    start = get_tsc();

    Sleep(100); // Aguarde 100 milissegundos

    end = get_tsc();
    QueryPerformanceCounter(&end_time);

    double elapsed_time = (double)(end_time.QuadPart - start_time.QuadPart) / freq.QuadPart;
    double cpu_frequency = (double)(end - start) / (elapsed_time * 1e6); // Frequência em MHz
    return cpu_frequency;
}

// Multiplicação de matrizes SEM OpenMP
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

// Multiplicação de matrizes COM OpenMP
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
    // Configurar ambiente de localização para suportar caracteres especiais
    setlocale(LC_ALL, "Portuguese_Brazil.1252");

    int tamanhos[] = {512, 1024, 2048};
    int num_tamanhos = sizeof(tamanhos) / sizeof(tamanhos[0]);

    for (int idx = 0; idx < num_tamanhos; idx++) {
        int N = tamanhos[idx];
        printf("\nExecutando para N = %d\n", N);

        double **A = (double **)malloc(N * sizeof(double *));
        double **B = (double **)malloc(N * sizeof(double *));
        double **C = (double **)malloc(N * sizeof(double *));
        for (int i = 0; i < N; i++) {
            A[i] = (double *)malloc(N * sizeof(double));
            B[i] = (double *)malloc(N * sizeof(double));
            C[i] = (double *)malloc(N * sizeof(double));
        }

        double total_time_seq = 0, total_time_omp = 0;

        for (int t = 0; t < NUM_TESTES; t++) {
            srand(1); // Semente fixa para números aleatórios
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    A[i][j] = B[i][j] = rand() % 100;

            LARGE_INTEGER start, end, freq;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            multiply_matrices_seq(A, B, C, N);
            QueryPerformanceCounter(&end);
            double exec_time = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
            double cpu_frequency = calculate_cpu_frequency();
            printf("Teste %d (Sem OpenMP): %f segundos | Memoria: %.2f MB | CPU: %.2f%% | Frequencia: %.2f MHz\n", 
                   t + 1, exec_time, get_memory_usage(), get_cpu_usage(), cpu_frequency);
            total_time_seq += exec_time;
        }

        for (int t = 0; t < NUM_TESTES; t++) {
            srand(1); // Semente fixa para números aleatórios
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    A[i][j] = B[i][j] = rand() % 100;

            LARGE_INTEGER start, end, freq;
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            multiply_matrices_omp(A, B, C, N);
            QueryPerformanceCounter(&end);
            double exec_time = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
            double cpu_frequency = calculate_cpu_frequency();
            printf("Teste %d (Com OpenMP): %f segundos | Memoria: %.2f MB | CPU: %.2f%% | Frequencia: %.2f MHz\n", 
                   t + 1, exec_time, get_memory_usage(), get_cpu_usage(), cpu_frequency);
            total_time_omp += exec_time;
        }

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
