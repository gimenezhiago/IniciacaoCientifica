#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <omp.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>

// Variáveis globais
int N;
double **A, **B, **C;
int num_threads;
int block_size;

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Aloca matriz bidimensional
double **allocate_matrix(int size) {
    double **matrix = new double*[size];
    for (int i = 0; i < size; i++)
        matrix[i] = new double[size]();
    return matrix;
}

// Libera matriz bidimensional
void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++)
        delete[] matrix[i];
    delete[] matrix;
}

// Soma A + B -> C
void add(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] + B[i][j];
}

// Subtrai A - B -> C
void subtract(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            C[i][j] = A[i][j] - B[i][j];
}

// Multiplicação com bloqueio usando Eigen + OpenMP
void multiply_blocking(double **A, double **B, double **C, int size, int block_size) {
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int ii = 0; ii < size; ii += block_size) {
        for (int jj = 0; jj < size; jj += block_size) {
            for (int kk = 0; kk < size; kk += block_size) {
                int M = std::min(block_size, size - ii);
                int N_ = std::min(block_size, size - jj);
                int K = std::min(block_size, size - kk);

                std::vector<double> A_block(M * K);
                std::vector<double> B_block(K * N_);
                std::vector<double> C_block(M * N_, 0.0);

                // Copia blocos para buffers 1D
                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++)
                        A_block[i * K + k] = A[ii + i][kk + k];

                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N_; j++)
                        B_block[k * N_ + j] = B[kk + k][jj + j];

                // Mapeia blocos com Eigen
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A_block.data(), M, K);
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B_block.data(), K, N_);
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C_block.data(), M, N_);

                C_eigen.noalias() += A_eigen * B_eigen;

                // Acumula resultado no C global
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N_; j++)
                        #pragma omp atomic
                        C[ii + i][jj + j] += C_block[i * N_ + j];
            }
        }
    }
}

// Strassen recursivo com OpenMP
void strassen_omp(double **A, double **B, double **C, int size) {
    if (size <= block_size) {
        multiply_blocking(A, B, C, size, block_size);
        return;
    }

    int newSize = size / 2;

    // Aloca submatrizes
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

    // Divide matrizes
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

    // Computa M1..M7 em paralelo
    #pragma omp parallel sections num_threads(num_threads)
    {
        #pragma omp section
        { add(A11, A22, AResult, newSize); add(B11, B22, BResult, newSize); strassen_omp(AResult, BResult, M1, newSize); }
        #pragma omp section
        { add(A21, A22, AResult, newSize); strassen_omp(AResult, B11, M2, newSize); }
        #pragma omp section
        { subtract(B12, B22, BResult, newSize); strassen_omp(A11, BResult, M3, newSize); }
        #pragma omp section
        { subtract(B21, B11, BResult, newSize); strassen_omp(A22, BResult, M4, newSize); }
        #pragma omp section
        { add(A11, A12, AResult, newSize); strassen_omp(AResult, B22, M5, newSize); }
        #pragma omp section
        { subtract(A21, A11, AResult, newSize); add(B11, B12, BResult, newSize); strassen_omp(AResult, BResult, M6, newSize); }
        #pragma omp section
        { subtract(A12, A22, AResult, newSize); add(B21, B22, BResult, newSize); strassen_omp(AResult, BResult, M7, newSize); }
    }

    double **temp1 = allocate_matrix(newSize);
    double **temp2 = allocate_matrix(newSize);

    // Combina resultados
    add(M1, M4, temp1, newSize);
    subtract(temp1, M5, temp2, newSize);
    add(temp2, M7, C11, newSize);

    add(M3, M5, C12, newSize);
    add(M2, M4, C21, newSize);

    subtract(M1, M2, temp1, newSize);
    add(temp1, M3, temp2, newSize);
    add(temp2, M6, C22, newSize);

    // Junta quadrantes
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }

    // Libera tudo
    free_matrix(A11, newSize); free_matrix(A12, newSize);
    free_matrix(A21, newSize); free_matrix(A22, newSize);
    free_matrix(B11, newSize); free_matrix(B12, newSize);
    free_matrix(B21, newSize); free_matrix(B22, newSize);
    free_matrix(C11, newSize); free_matrix(C12, newSize);
    free_matrix(C21, newSize); free_matrix(C22, newSize);
    free_matrix(M1, newSize); free_matrix(M2, newSize);
    free_matrix(M3, newSize); free_matrix(M4, newSize);
    free_matrix(M5, newSize); free_matrix(M6, newSize);
    free_matrix(M7, newSize); free_matrix(AResult, newSize);
    free_matrix(BResult, newSize); free_matrix(temp1, newSize);
    free_matrix(temp2, newSize);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <tamanho_matriz> <num_threads> <block_size>\n";
        return 1;
    }

    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    block_size = atoi(argv[3]);

    if ((N & (N - 1)) != 0) {
        std::cerr << "Erro: o tamanho da matriz deve ser potência de 2.\n";
        return 1;
    }

    A = allocate_matrix(N);
    B = allocate_matrix(N);
    C = allocate_matrix(N);

    srand(1);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }

    double start = get_time();
    strassen_omp(A, B, C, N);
    double end = get_time();

    std::cout << "Strassen + OpenMP + Eigen (block_size = " << block_size
              << ", threads = " << num_threads << "): "
              << (end - start) << " s\n";

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);

    return 0;
}
