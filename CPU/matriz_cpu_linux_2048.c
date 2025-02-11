#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <pthread.h>
#include <omp.h>
#include <sys/sysinfo.h>

#define N 2048
#define NUM_THREADS_1 4
#define NUM_THREADS_2 8
#define NUM_THREADS_3 16

double A[N][N], B[N][N], C[N][N];

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Função para obter uso de memória em MB
double get_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

// Função para obter uso de CPU em %
double get_cpu_usage() {
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return 0.0;
    long user, nice, system, idle;
    fscanf(fp, "cpu %ld %ld %ld %ld", &user, &nice, &system, &idle);
    fclose(fp);
    return 100.0 * (user + nice + system) / (user + nice + system + idle);
}

// Função sequencial para multiplicação de matrizes
void multiply_matrices_seq() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Função para multiplicação de matrizes com OpenMP
void multiply_matrices_omp(int num_threads) {
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

// Estrutura de dados para Pthreads
typedef struct {
    int start, end;
} ThreadData;

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

void run_pthread(int num_threads) {
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

int main() {
    srand(1);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = rand() % 100;

    double start, end;

    // Teste Sequencial
    start = get_time();
    multiply_matrices_seq();
    end = get_time();
    printf("Sequencial: Tempo = %.4f s | Memória = %.2f MB | CPU = %.2f%%\n", end - start, get_memory_usage(), get_cpu_usage());
    
    // Teste OpenMP
    int threads[] = {NUM_THREADS_1, NUM_THREADS_2, NUM_THREADS_3};
    for (int i = 0; i < 3; i++) {
        start = get_time();
        multiply_matrices_omp(threads[i]);
        end = get_time();
        printf("OpenMP (%d threads): Tempo = %.4f s | Memória = %.2f MB | CPU = %.2f%%\n", threads[i], end - start, get_memory_usage(), get_cpu_usage());
    }
    
    // Teste Pthreads
    for (int i = 0; i < 3; i++) {
        start = get_time();
        run_pthread(threads[i]);
        end = get_time();
        printf("Pthreads (%d threads): Tempo = %.4f s | Memória = %.2f MB | CPU = %.2f%%\n", threads[i], end - start, get_memory_usage(), get_cpu_usage());
    }
    
    return 0;
}
