#include <cuda_runtime.h>

__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);


  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
  //hostFunction(a, N);
  deviceKernel<<<blocksPerGrid, threadsPerBlock>>>(a, N);
  cudaDeviceSynchronize();
  cudaFree(a);
}


