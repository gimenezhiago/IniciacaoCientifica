#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

// Variáveis globais
int N;
double **A, **B, **C;
int num_threads = N

typedef struct {
    double **A;
    double **B;
    double **C;
    int size;
} ThreadData;

// Tempo
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Utilitários
double **allocate_matrix(int size) {
    double **matrix = malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
        matrix[i] = malloc(size * sizeof(double));
    return matrix;
}

void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) free(matrix[i]);
    free(matrix);
}

void add(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void subtract(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

// Strassen recursivo
void strassen(double **A, double **B, double **C, int size);

void *thread_strassen(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    strassen(data->A, data->B, data->C, data->size);
    return NULL;
}

void strassen(double **A, double **B, double **C, int size) {
    if (size <= 2) {
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

    // Alocar resultados
    double **M1 = allocate_matrix(newSize);
    double **M2 = allocate_matrix(newSize);
    double **M3 = allocate_matrix(newSize);
    double **M4 = allocate_matrix(newSize);
    double **M5 = allocate_matrix(newSize);
    double **M6 = allocate_matrix(newSize);
    double **M7 = allocate_matrix(newSize);
    double **AResult = allocate_matrix(newSize);
    double **BResult = allocate_matrix(newSize);

    pthread_t threads[7];
    ThreadData tdata[7];

    // M1 = (A11 + A22) * (B11 + B22)
    add(A11, A22, AResult, newSize);
    add(B11, B22, BResult, newSize);
    tdata[0] = (ThreadData){AResult, BResult, M1, newSize};

    // M2 = (A21 + A22) * B11
    add(A21, A22, AResult, newSize);
    tdata[1] = (ThreadData){AResult, B11, M2, newSize};

    // M3 = A11 * (B12 - B22)
    subtract(B12, B22, BResult, newSize);
    tdata[2] = (ThreadData){A11, BResult, M3, newSize};

    // M4 = A22 * (B21 - B11)
    subtract(B21, B11, BResult, newSize);
    tdata[3] = (ThreadData){A22, BResult, M4, newSize};

    // M5 = (A11 + A12) * B22
    add(A11, A12, AResult, newSize);
    tdata[4] = (ThreadData){AResult, B22, M5, newSize};

    // M6 = (A21 - A11) * (B11 + B12)
    subtract(A21, A11, AResult, newSize);
    add(B11, B12, BResult, newSize);
    tdata[5] = (ThreadData){AResult, BResult, M6, newSize};

    // M7 = (A12 - A22) * (B21 + B22)
    subtract(A12, A22, AResult, newSize);
    add(B21, B22, BResult, newSize);
    tdata[6] = (ThreadData){AResult, BResult, M7, newSize};

    for (int i = 0; i < 7; i++)
        pthread_create(&threads[i], NULL, thread_strassen, &tdata[i]);

    for (int i = 0; i < 7; i++)
        pthread_join(threads[i], NULL);

    double **C11 = allocate_matrix(newSize);
    double **C12 = allocate_matrix(newSize);
    double **C21 = allocate_matrix(newSize);
    double **C22 = allocate_matrix(newSize);
    double **temp1 = allocate_matrix(newSize);
    double **temp2 = allocate_matrix(newSize);

    add(M1, M4, temp1, newSize);
    subtract(temp1, M5, temp2, newSize);
    add(temp2, M7, C11, newSize);

    add(M3, M5, C12, newSize);
    add(M2, M4, C21, newSize);

    subtract(M1, M2, temp1, newSize);
    add(temp1, M3, temp2, newSize);
    add(temp2, M6, C22, newSize);

    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }

    // Limpeza
    free_matrix(A11, newSize); free_matrix(A12, newSize);
    free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize);
    free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(M1, newSize); free_matrix(M2, newSize);
    free_matrix(M3, newSize); free_matrix(M4, newSize);
    free_matrix(M5, newSize); free_matrix(M6, newSize);
    free_matrix(M7, newSize); free_matrix(AResult, newSize);
    free_matrix(BResult, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize);
    free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(temp1, newSize); free_matrix(temp2, newSize);
}

// Função principal
int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho_matriz>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    if ((N & (N - 1)) != 0) {
        printf("Erro: o tamanho da matriz deve ser potência de 2.\n");
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
    strassen(A, B, C, N);
    double end = get_time();
    printf("Strassen + Pthreads: Tempo = %.4f s\n", end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
