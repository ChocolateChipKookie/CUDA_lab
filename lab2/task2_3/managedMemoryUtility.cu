#include <stdio.h>
#include <cuda_runtime.h>
#include <string>

__global__
void vectorAdd(const float *A, const float *B, const float *C, int numElements){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc char* argv[]){
	cudaError_t err = cudaSuccess;
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	
	int dev;
	cudaGetDevice(&dev);
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	if(size>=deviceProp.totalGlobalMem)
	{
		printf("The data set is too large");
		exit(EXIT_FAILURE);
	}
	
	// Allocate the device input vector A                                                                                                                         
	float *d_A = NULL;
    err = cudaMallocManaged(&d_A,size);

    if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMallocManaged(&d_B,size);

    if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	// Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMallocManaged(&d_C,size);

    if (err != cudaSuccess)                                                                                                                                           {
                fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

    // Verify that allocations succeeded
    if (d_A == NULL || d_B == NULL || d_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        d_A[i] = rand()/(float)RAND_MAX;
        d_B[i] = rand()/(float)RAND_MAX;
    }
	
	// Launch the Vector Add CUDA Kernel

    //int threadsPerBlock = std::atoi(argv[1]);
    int threadsPerBlock = 32;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(d_A[i] + d_B[i] - d_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");
	
	// Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);                                                                                                                                       }

    printf("Done\n");
    return 0;
}