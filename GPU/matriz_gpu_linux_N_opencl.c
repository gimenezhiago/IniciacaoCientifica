#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>

#define N 512

const char *kernelSource =
"__kernel void matrixMul(__global double *A, __global double *B, __global double *C, int n) {" 
"    int row = get_global_id(1);"
"    int col = get_global_id(0);"
"    if (row < n && col < n) {"
"        double sum = 0.0;"
"        for (int k = 0; k < n; k++) {"
"            sum += A[row * n + k] * B[k * n + col];"
"        }"
"        C[row * n + col] = sum;"
"    }"
"}";

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

int main() {
    size_t size = N * N * sizeof(double);
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }
    
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_A, d_B, d_C;
    
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    
    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size, h_A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, size, h_B, 0, NULL, NULL);
    
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrixMul", NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    
    size_t globalSize[2] = {N, N};
    
    double start = get_time();
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);
    double end = get_time();
    
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL);
    
    printf("Tem %d\n", globalSize[2]);
    printf("Matriz de %d x %d\n", N, N);
    printf("OpenCL: Tempo = %.4f s\n", end - start);
    
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
