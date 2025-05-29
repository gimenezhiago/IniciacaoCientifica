#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <sys/time.h>
#include <pthread.h>

using namespace Eigen;

// Variáveis globais
int N, block_size;
int num_threads;
MatrixXd A, B, C;

// Estrutura de dados para Pthreads
typedef struct {
    int start, end;
} ThreadData;

// Função para obter tempo atual em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Função para multiplicação de matrizes com Pthreads e Eigen
void *multiply_matrices_pthread_eigen(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    for (int i = data->start; i < data->end; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                int i_max = std::min(i + block_size, data->end);
                int j_max = std::min(j + block_size, N);
                int k_max = std::min(k + block_size, N);
                
                // Usa Eigen para multiplicar blocos
                C.block(i, j, i_max-i, j_max-j).noalias() += 
                    A.block(i, k, i_max-i, k_max-k) * B.block(k, j, k_max-k, j_max-j);
            }
        }
    }

    return NULL;
}

void run_pthread_eigen() {
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int chunk = N / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start = i * chunk;
        thread_data[i].end = (i == num_threads - 1) ? N : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, multiply_matrices_pthread_eigen, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
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

    // Configuração inicial do Eigen
    Eigen::initParallel();
    
    // Alocação das matrizes usando Eigen
    A = MatrixXd::Random(N, N);
    B = MatrixXd::Random(N, N);
    C = MatrixXd::Zero(N, N);

    // Inicialização com valores controlados
    A = (A.array().abs() * 100).cast<double>();
    B = (B.array().abs() * 100).cast<double>();

    double start = get_time();
    run_pthread_eigen();
    double end = get_time();

    std::cout << "Pthreads + Eigen (" << num_threads << " threads, bloco " << block_size 
              << "): Tempo = " << end - start << " s\n";

    return 0;
}