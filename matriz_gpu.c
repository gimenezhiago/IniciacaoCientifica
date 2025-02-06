#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024  // Tamanho da matriz
#define NUM_TESTES 10  // Número de repetições para cálculo da média

__global__ void multiply_matrices(double *A, double *B, double *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    double *A, *B, *C;
    double *d_A, *d_B, *d_C;

    size_t size = N * N * sizeof(double);

    A = (double *)malloc(size);
    B = (double *)malloc(size);
    C = (double *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = B[i] = rand() % 100;
    }

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(N / blockSize.x, N / blockSize.y);

    float total_time = 0;

    for (int t = 0; t < NUM_TESTES; t++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        multiply_matrices<<<gridSize, blockSize>>>(d_A, d_B, d_C);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        printf("Teste %d: %f segundos\n", t + 1, milliseconds / 1000);
    }

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Tempo médio de execução na GPU: %f segundos\n", (total_time / NUM_TESTES) / 1000);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
