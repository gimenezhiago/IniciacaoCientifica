#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <sys/time.h>
#include <pthread.h>
#include <algorithm>

using namespace Eigen;

// Variáveis globais
int N, block_size;
int num_threads;
MatrixXd A, B, C;

// Estrutura para as informações da thread
struct ThreadData {
    int start, end;
};

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Função executada por cada thread (com blocos + Eigen)
void* multiply_matrices_pthread_eigen(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    for (int i = data->start; i < data->end; i += block_size) {
        int i_max = std::min(i + block_size, data->end);
        for (int j = 0; j < N; j += block_size) {
            int j_max = std::min(j + block_size, N);
            for (int k = 0; k < N; k += block_size) {
                int k_max = std::min(k + block_size, N);

                // Multiplicação de blocos com Eigen
                C.block(i, j, i_max - i, j_max - j).noalias() +=
                    A.block(i, k, i_max - i, k_max - k) *
                    B.block(k, j, k_max - k, j_max - j);
            }
        }
    }

    return nullptr;
}

// Função que cria e gerencia threads Pthreads
void run_pthread_eigen() {
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> thread_data(num_threads);

    int chunk = N / num_threads;
    int remainder = N % num_threads;

    int current_row = 0;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].start = current_row;
        thread_data[i].end = current_row + chunk + (i < remainder ? 1 : 0);
        current_row = thread_data[i].end;

        if (pthread_create(&threads[i], nullptr, multiply_matrices_pthread_eigen, &thread_data[i]) != 0) {
            std::cerr << "Erro ao criar a thread " << i << std::endl;
            exit(1);
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <tamanho_matriz> <num_threads> <blocking_size>\n";
        return 1;
    }

    N = std::atoi(argv[1]);
    num_threads = std::atoi(argv[2]);
    block_size = std::atoi(argv[3]);

    if ((N & (N - 1)) != 0) {
        std::cerr << "Erro: o tamanho da matriz deve ser potência de 2.\n";
        return 1;
    }

    // Inicializa Eigen (opcional para multithread interno)
    Eigen::initParallel();

    // Cria e inicializa matrizes
    A = MatrixXd::Random(N, N).cwiseAbs() * 100;
    B = MatrixXd::Random(N, N).cwiseAbs() * 100;
    C = MatrixXd::Zero(N, N);

    double start = get_time();
    run_pthread_eigen();
    double end = get_time();

    std::cout << "Pthreads + Eigen (" << num_threads << " threads, bloco " << block_size
              << "): Tempo = " << (end - start) << " s\n";

    return 0;
}
