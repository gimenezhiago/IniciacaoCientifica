#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

// Variáveis globais
int N;
double **A, **B, **C;
int num_threads;

typedef struct {
    int start, end;
} ThreadData;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

double **allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
        matrix[i] = (double *)calloc(size, sizeof(double));
    return matrix;
}

void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++)
        free(matrix[i]);
    free(matrix);
}

void add_matrix(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void sub_matrix(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void strassen(double **A, double **B, double **C, int size) {
    if (size <= 64) {
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                C[i][j] = 0;
                for (int k = 0; k < size; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
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

    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }

    strassen(A11, B11, M1, newSize); // M1 = A11 * B11
    strassen(A12, B21, M2, newSize); // M2 = A12 * B21
    strassen(A11, B12, M3, newSize); // M3 = A11 * B12
    strassen(A12, B22, M4, newSize); // M4 = A12 * B22
    strassen(A21, B11, M5, newSize); // M5 = A21 * B11
    strassen(A22, B21, M6, newSize); // M6 = A22 * B21
    strassen(A21, B12, M7, newSize); // M7 = A21 * B12

    // C11 = M1 + M2
    add_matrix(M1, M2, C11, newSize);
    // C12 = M3 + M4
    add_matrix(M3, M4, C12, newSize);
    // C21 = M5 + M6
    add_matrix(M5, M6, C21, newSize);
    // C22 = M3 + M7
    add_matrix(M3, M7, C22, newSize); // apenas exemplo de variação, pode mudar

    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }

    free_matrix(A11, newSize); free_matrix(A12, newSize); free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize); free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize); free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(M1, newSize); free_matrix(M2, newSize); free_matrix(M3, newSize); free_matrix(M4, newSize);
    free_matrix(M5, newSize); free_matrix(M6, newSize); free_matrix(M7, newSize); free_matrix(T1, newSize); free_matrix(T2, newSize);
}

// Thread executa multiplicação de parte de A com B usando Strassen
void *multiply_strassen_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    strassen(A + data->start, B, C + data->start, N);
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
    if (argc != 3) {
        printf("Uso: %s <tamanho_matriz> <num_threads>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    if ((N & (N - 1)) != 0) {
        printf("Erro: o tamanho da matriz deve ser potência de 2 (ex: 256, 512...)\n");
        return 1;
    }

    A = allocate_matrix(N);
    B = allocate_matrix(N);
    C = allocate_matrix(N);

    srand(1);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = rand() % 100;

    double start = get_time();
    run_pthread();
    double end = get_time();

    printf("Strassen com Pthreads (%d threads): Tempo = %.4f s\n", num_threads, end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
