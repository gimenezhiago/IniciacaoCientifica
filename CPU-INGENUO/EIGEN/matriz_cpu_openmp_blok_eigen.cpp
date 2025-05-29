#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <sys/time.h>
#include <omp.h>

using namespace Eigen;

// Variáveis globais
int N, block_size;
int num_threads;

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Função para multiplicação de matrizes com OpenMP e Eigen
void multiply_matrices_omp_eigen(MatrixXd& A, MatrixXd& B, MatrixXd& C) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                int i_max = std::min(i + block_size, N);
                int j_max = std::min(j + block_size, N);
                int k_max = std::min(k + block_size, N);
                
                // Usa Eigen para multiplicar blocos
                C.block(i, j, i_max-i, j_max-j).noalias() += 
                    A.block(i, k, i_max-i, k_max-k) * B.block(k, j, k_max-k, j_max-j);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "Uso: " << argv[0] << " <tamanho_matriz> <num_threads> <blocking_size>\n";
        return 1;
    }

    N = atoi(argv[1]);
    num_threads = atoi(argv[2]);
    block_size = atoi(argv[3]);

    // Configura Eigen para usar o número especificado de threads
    Eigen::setNbThreads(num_threads);
    Eigen::initParallel();

    // Alocação das matrizes usando Eigen
    MatrixXd A = MatrixXd::Random(N, N);
    MatrixXd B = MatrixXd::Random(N, N);
    MatrixXd C = MatrixXd::Zero(N, N);

    // Inicialização das matrizes com valores controlados
    A = (A.array().abs() * 100).cast<double>();
    B = (B.array().abs() * 100).cast<double>();

    double start = get_time();
    multiply_matrices_omp_eigen(A, B, C);
    double end = get_time();

    std::cout << "OpenMP + Eigen (" << num_threads << " threads, bloco " << block_size 
              << "): Tempo = " << end - start << " s\n";

    return 0;
}