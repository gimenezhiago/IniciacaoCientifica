#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Variáveis globais
int N;
double **A, **B, **C;

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Função para alocar matriz NxN
double **allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

// Função para liberar matriz NxN
void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Soma de matrizes: C = A + B
void add(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

// Subtração de matrizes: C = A - B
void subtract(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

// Algoritmo de Strassen
void strassen(double **A, double **B, double **C, int size) {
    if (size <= 2) {
        // Caso base: multiplicação ingênua para blocos 2x2 ou menores
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                C[i][j] = 0;
                for (int k = 0; k < size; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
        return;
    }

    int newSize = size / 2;

    // Alocar submatrizes
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

    double **AResult = allocate_matrix(newSize);
    double **BResult = allocate_matrix(newSize);

    // Dividir as matrizes em 4 submatrizes
    for (int i = 0; i < newSize; i++) {
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
    }

    // M1 = (A11 + A22) * (B11 + B22)
    add(A11, A22, AResult, newSize);
    add(B11, B22, BResult, newSize);
    strassen(AResult, BResult, M1, newSize);

    // M2 = (A21 + A22) * B11
    add(A21, A22, AResult, newSize);
    strassen(AResult, B11, M2, newSize);

    // M3 = A11 * (B12 - B22)
    subtract(B12, B22, BResult, newSize);
    strassen(A11, BResult, M3, newSize);

    // M4 = A22 * (B21 - B11)
    subtract(B21, B11, BResult, newSize);
    strassen(A22, BResult, M4, newSize);

    // M5 = (A11 + A12) * B22
    add(A11, A12, AResult, newSize);
    strassen(AResult, B22, M5, newSize);

    // M6 = (A21 - A11) * (B11 + B12)
    subtract(A21, A11, AResult, newSize);
    add(B11, B12, BResult, newSize);
    strassen(AResult, BResult, M6, newSize);

    // M7 = (A12 - A22) * (B21 + B22)
    subtract(A12, A22, AResult, newSize);
    add(B21, B22, BResult, newSize);
    strassen(AResult, BResult, M7, newSize);

    // C11 = M1 + M4 - M5 + M7
    add(M1, M4, AResult, newSize);
    subtract(AResult, M5, BResult, newSize);
    add(BResult, M7, C11, newSize);

    // C12 = M3 + M5
    add(M3, M5, C12, newSize);

    // C21 = M2 + M4
    add(M2, M4, C21, newSize);

    // C22 = M1 - M2 + M3 + M6
    subtract(M1, M2, AResult, newSize);
    add(AResult, M3, BResult, newSize);
    add(BResult, M6, C22, newSize);

    // Juntar as submatrizes em C
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }

    // Liberar memória
    free_matrix(A11, newSize); free_matrix(A12, newSize);
    free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize);
    free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize);
    free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(M1, newSize); free_matrix(M2, newSize);
    free_matrix(M3, newSize); free_matrix(M4, newSize);
    free_matrix(M5, newSize); free_matrix(M6, newSize);
    free_matrix(M7, newSize);
    free_matrix(AResult, newSize); free_matrix(BResult, newSize);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho_matriz>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    // Para Strassen, N deve ser potência de 2
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

    printf("Strassen: Tempo = %.4f s\n", end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
