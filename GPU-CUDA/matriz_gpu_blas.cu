#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Função para obter o tempo atual
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho_matriz>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(double);

    // Alocação e inicialização no host
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    srand(1);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = h_B[i] = rand() % 100;
        h_C[i] = 0.0;
    }

    // Alocação na GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copiar para a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Criação do handle cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Parâmetros de alpha e beta
    double alpha = 1.0, beta = 0.0;

    // Execução com cuBLAS
    double start = get_time();
    // cublasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    double end = get_time();

    // Copiar o resultado de volta para o host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("cuBLAS: Tempo = %.4f s\n", end - start);

    // Liberar memória
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
