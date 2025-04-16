#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Kernel CUDA simples — uma thread por elemento da matriz C
__global__ void matrixMulSimple(double *A, double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Índice da linha
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Índice da coluna

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho_matriz>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(double);

    // Alocação no host
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    // Inicialização
    srand(1);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = h_B[i] = rand() % 100;
    }

    // Alocação na GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Cópia para a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Definir número de threads e blocos (grid 2D)
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15) / 16, (N + 15) / 16);

    // Execução e tempo
    double start = get_time();
    matrixMulSimple<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double end = get_time();

    printf("CUDA (Sem Blocking/SIMD): Tempo = %.4f s\n", end - start);

    // Cópia de volta para a CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Liberação
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
