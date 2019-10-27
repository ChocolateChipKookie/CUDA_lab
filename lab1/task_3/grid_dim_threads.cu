#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void) {
  printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
  blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}

__global__ void
vectorAdd(int *A, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        A[i] = A[i] * 2;
    }
}

int main(int argc, char **argv) {

  int nElem = 50000;
  int *h_vect = (int *)malloc(nElem * sizeof(int));
  int *d_vect = NULL;

  for (int i = 0; i < nElem; ++i)
     {
         h_vect[i] = rand();
     }

  cudaMalloc((void **)&d_vect, nElem * sizeof(int));

  int threadsPerBlock = 256;
  int blocksPerGrid = (nElem+threadsPerBlock-1)/threadsPerBlock;
  // check grid and block dimension from host side
  cudaMemcpy(d_vect, h_vect, nElem * sizeof(int), cudaMemcpyHostToDevice);

  vectorAdd<<<threadsPerBlock, blocksPerGrid>>>(d_vect, nElem);
  cudaDeviceSynchronize();

  cudaMemcpy(h_vect, d_vect, nElem * sizeof(int), cudaMemcpyDeviceToHost);

  // check grid and block dimension from device side
  //checkIndex <<<grid, block>>> ();
  cudaDeviceSynchronize();
  // reset device before you leave
  cudaDeviceReset();
  free(h_vect);
  cudaFree(d_vect);
  return(0);
}