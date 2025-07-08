#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <cblas.h>  // ✅ Para BLAS

int N;
int block_size;

double **A, **B, **C;

typedef struct {
    double **A, **B, **C;
    int size;
} ThreadData;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

double **allocate_matrix(int size) {
    double **matrix = malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = calloc(size, sizeof(double));
    }
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

// ✅ Multiplicação com Blocking + BLAS (sem SIMD)
void multiply_blocking(double **A, double **B, double **C, int size, int block_size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = 0.0;

    for (int ii = 0; ii < size; ii += block_size) {
        for (int jj = 0; jj < size; jj += block_size) {
            for (int kk = 0; kk < size; kk += block_size) {

                int M = (ii + block_size > size) ? size - ii : block_size;
                int N = (jj + block_size > size) ? size - jj : block_size;
                int K = (kk + block_size > size) ? size - kk : block_size;

                // BLAS espera vetores contínuos. Copie blocos para buffers:
                double *Ablock = malloc(M * K * sizeof(double));
                double *Bblock = malloc(K * N * sizeof(double));
                double *Cblock = calloc(M * N, sizeof(double));

                // Copia blocos A[ii:ii+M][kk:kk+K]
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++)
                        Ablock[i*K + k] = A[ii + i][kk + k];

                // Copia blocos B[kk:kk+K][jj:jj+N]
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N; j++)
                        Bblock[k*N + j] = B[kk + k][jj + j];

                // Multiplica com cblas_dgemm
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0, Ablock, K, Bblock, N, 1.0, Cblock, N);

                // Acumula resultado em C[ii:ii+M][jj:jj+N]
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                        C[ii + i][jj + j] += Cblock[i*N + j];

                free(Ablock);
                free(Bblock);
                free(Cblock);
            }
        }
    }
}

void strassen(double **A, double **B, double **C, int size);

void *thread_strassen(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    strassen(data->A, data->B, data->C, data->size);
    return NULL;
}

void strassen(double **A, double **B, double **C, int size) {
    if (size <= block_size) {
        multiply_blocking(A, B, C, size, block_size);
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

    double **M[7], **AResult[7], **BResult[7];
    pthread_t threads[7];
    ThreadData tdata[7];

    for (int i = 0; i < 7; i++) {
        M[i] = allocate_matrix(newSize);
        AResult[i] = allocate_matrix(newSize);
        BResult[i] = allocate_matrix(newSize);
    }

    add(A11, A22, AResult[0], newSize);
    add(B11, B22, BResult[0], newSize);
    tdata[0] = (ThreadData){AResult[0], BResult[0], M[0], newSize};

    add(A21, A22, AResult[1], newSize);
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++)
            BResult[1][i][j] = B11[i][j];
    tdata[1] = (ThreadData){AResult[1], BResult[1], M[1], newSize};

    subtract(B12, B22, BResult[2], newSize);
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++)
            AResult[2][i][j] = A11[i][j];
    tdata[2] = (ThreadData){AResult[2], BResult[2], M[2], newSize};

    subtract(B21, B11, BResult[3], newSize);
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++)
            AResult[3][i][j] = A22[i][j];
    tdata[3] = (ThreadData){AResult[3], BResult[3], M[3], newSize};

    add(A11, A12, AResult[4], newSize);
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++)
            BResult[4][i][j] = B22[i][j];
    tdata[4] = (ThreadData){AResult[4], BResult[4], M[4], newSize};

    subtract(A21, A11, AResult[5], newSize);
    add(B11, B12, BResult[5], newSize);
    tdata[5] = (ThreadData){AResult[5], BResult[5], M[5], newSize};

    subtract(A12, A22, AResult[6], newSize);
    add(B21, B22, BResult[6], newSize);
    tdata[6] = (ThreadData){AResult[6], BResult[6], M[6], newSize};

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

    add(M[0], M[3], temp1, newSize);
    subtract(temp1, M[4], temp2, newSize);
    add(temp2, M[6], C11, newSize);

    add(M[2], M[4], C12, newSize);
    add(M[1], M[3], C21, newSize);

    subtract(M[0], M[1], temp1, newSize);
    add(temp1, M[2], temp2, newSize);
    add(temp2, M[5], C22, newSize);

    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }

    for (int i = 0; i < 7; i++) {
        free_matrix(M[i], newSize);
        free_matrix(AResult[i], newSize);
        free_matrix(BResult[i], newSize);
    }
    free_matrix(A11, newSize); free_matrix(A12, newSize);
    free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize);
    free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize);
    free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(temp1, newSize); free_matrix(temp2, newSize);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <tamanho_matriz> <block_size>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    block_size = atoi(argv[2]);

    if ((N & (N - 1)) != 0) {
        printf("Erro: tamanho deve ser potência de 2.\n");
        return 1;
    }

    A = allocate_matrix(N);
    B = allocate_matrix(N);
    C = allocate_matrix(N);

    srand(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = B[i][j] = rand() % 10;

    double start = get_time();
    strassen(A, B, C, N);
    double end = get_time();
    printf("Strassen Pthreads + Blocking + BLAS (block_size = %d): %.4f s\n", block_size, end - start);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
