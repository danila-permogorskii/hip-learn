#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#define N 32
#define BLOCK_SIZE 16

// a and b are two matrices
// c is a matrix product
// n is a matrix size
void cpu_matrix_multiplication(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++) {
        sum += a[i * n + k] * b[k * n + j];
      }
      c[i * n + j] = sum;
    }
  }
}

__global__ void gpu_matrix_multiplication(float *a, float *b, float *c, int n) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  if (row < n && col < n) {
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

int main() {
  std::cout << "Everything is ok" << std::endl;

  float *a, *b, *c;

  a = (float *)malloc(sizeof(float) * N * N);
  b = (float *)malloc(sizeof(float) * N * N);
  c = (float *)malloc(sizeof(float) * N * N);

  // Initialise matrix A
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  // Initialise matrix B
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      b[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  cpu_matrix_multiplication(a, b, c, N);

  free(a);
  free(b);
  free(c);

  float *h_a, *h_b, *h_c;
  h_a = (float *)malloc(sizeof(float) * N * N);
  h_b = (float *)malloc(sizeof(float) * N * N);
  h_c = (float *)malloc(sizeof(float) * N * N);

  // initialise matrix A
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_a[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }
  // initialise matrix B
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_b[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  float *d_a, *d_b, *d_c;
  hipMalloc((void **)&d_a, sizeof(float) * N * N);
  hipMalloc((void **)&d_b, sizeof(float) * N * N);
  hipMalloc((void **)&d_c, sizeof(float) * N * N);

  hipMemcpy(d_a, h_a, sizeof(float) * N * N, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, sizeof(float) * N * N, hipMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  int n_blocks = static_cast<int>(ceil(static_cast<float>(N) / BLOCK_SIZE));
  dim3 blocksPerGrid(n_blocks, n_blocks);

  hipLaunchKernelGGL(gpu_matrix_multiplication, blocksPerGrid, threadsPerBlock,
                     0, 0, d_a, d_b, d_c, N);

  hipDeviceSynchronize();

  hipMemcpy(h_c, d_c, sizeof(float) * N * N, hipMemcpyDeviceToHost);

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  std::cout << "Everything is still ok" << std::endl;

  return 0;
}