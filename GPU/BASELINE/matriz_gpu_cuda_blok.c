#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Pode alterar se quiser testar outros blocos

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Kernel CUDA com blocking (tiling) e memória compartilhada
__global__ void matrixMulTiled(double *A, double *B, double *C, int N) {
    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double sum = 0.0;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < N && tile * TILE_SIZE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && tile * TILE_SIZE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
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

    // Inicialização das matrizes
    srand(1);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = h_B[i] = rand() % 100;
    }

    // Alocação na GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Cópia para GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid e blocos
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Execução e tempo
    double start = get_time();
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double end = get_time();

    printf("CUDA (Blocking %dx%d): Tempo = %.4f s\n", TILE_SIZE, TILE_SIZE, end - start);

    // Cópia de volta para a CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Liberação
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
