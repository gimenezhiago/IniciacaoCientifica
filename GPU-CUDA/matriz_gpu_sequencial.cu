#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

// Função para obter o tempo atual
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Kernel CUDA
__global__ void multiply_matrices_cuda(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <tamanho_matriz> <block_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int BLOCK_SIZE = atoi(argv[2]);

    size_t size = N * N * sizeof(double);

    // Alocação e inicialização no host
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    srand(1);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = h_B[i] = rand() % 100;
    }

    // Alocação no device
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Cópia para a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configuração da grid
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Execução
    double start = get_time();
    multiply_matrices_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double end = get_time();

    // Cópia do resultado de volta
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("CUDA: Tempo = %.4f s (BLOCK_SIZE = %d)\n", end - start, BLOCK_SIZE);

    // Liberação de memória
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
