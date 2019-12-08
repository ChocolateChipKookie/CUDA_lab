#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

using red_t = int;

__global__ void reduction_kernel(red_t* input, size_t size, size_t batch_size, unsigned steps)
{
	unsigned position = threadIdx.x * batch_size;
	unsigned step = 1;

	for(unsigned i = 0; i < steps; ++i)
	{
		if(position < size)
		{
			for (unsigned new_pos = position + step, j = 1; j < batch_size && new_pos < size; j++, new_pos += step)
			{
				input[position] += input[new_pos];
			}
		}
		position *= batch_size;
		step *= batch_size;
		__syncthreads();
	}
}


__global__ void reduction_kernel_shared(red_t* input, size_t size, size_t batch_size, unsigned steps)
{
	unsigned position = threadIdx.x * batch_size;
	unsigned step = 1;
	extern __shared__ red_t shared_memory[];

	if(steps > 1)
	{
		shared_memory[threadIdx.x] = 0;
		for (unsigned new_pos = position, j = 0; j < batch_size && new_pos < size; j++, new_pos += step)
		{
			shared_memory[threadIdx.x] += input[new_pos];
		}
		size = (size + 1 )/ batch_size;
		--steps;

		for (unsigned i = 0; i < steps; ++i)
		{
			if (position < size)
			{
				for (unsigned new_pos = position + step, j = 1; j < batch_size && new_pos < size; j++, new_pos += step)
				{
					shared_memory[position] += shared_memory[new_pos];
				}
			}
			position *= batch_size;
			step *= batch_size;
			__syncthreads();
		}
		if(position == 0)
		{
			input[0] = shared_memory[0];
		}
	}
	else
	{
		for (unsigned i = 0; i < steps; ++i)
		{
			if (position < size)
			{
				for (unsigned new_pos = position + step, j = 1; j < batch_size && new_pos < size; j++, new_pos += step)
				{
					input[position] += input[new_pos];
				}
			}
			position *= batch_size;
			step *= batch_size;
			__syncthreads();
		}
	}
}


red_t reduction_cuda(red_t* input_data, size_t size, size_t batch_size)
{
	red_t* dev_input = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(red_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_input, input_data, size * sizeof(red_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	size_t steps = 0;
	for(size_t tmp = 1; tmp < size; tmp *= batch_size, ++steps);

	size_t threads = size / batch_size + 1;

	if(threads > 1024)
	{
		throw std::runtime_error{ "Too many threads" };
	}

	dim3 threads_per_block(threads < 1024 ? threads : 1024);
	dim3 blocks_per_grid(1);
	reduction_kernel<<<blocks_per_grid, threads_per_block >>>(dev_input, size, batch_size, steps);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	red_t result;
	cudaStatus = cudaMemcpy(&result, dev_input, sizeof(red_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_input);

	return result;
}

red_t reduction_cuda_shared(red_t* input_data, size_t size, size_t batch_size)
{
	red_t* dev_input = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(red_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_input, input_data, size * sizeof(red_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	size_t steps = 0;
	for (size_t tmp = 1; tmp < size; tmp *= batch_size, ++steps);

	if (size / batch_size + 1 > 1024)
	{
		throw std::runtime_error{ "Batch size too small for this number of threads" };
	}
	dim3 threads_per_block(size / batch_size + 1);
	dim3 blocks_per_grid(1);
	reduction_kernel_shared<<<blocks_per_grid, threads_per_block, sizeof(red_t) * (size/batch_size + 1)>>> (dev_input, size, batch_size, steps);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	red_t result;
	cudaStatus = cudaMemcpy(&result, dev_input, sizeof(red_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_input);

	return result;
}


red_t test(red_t* input, size_t size)
{
	return std::accumulate(input, input + size, 0);
}

int main()
{
	std::default_random_engine engine;
	std::uniform_int_distribution<int> dist(0, 4);

	size_t size = 2048000;
	red_t* data = new red_t[size];

	for (size_t i = 0; i < size; ++i)
	{
		data[i] = dist(engine);
	}

	auto begin = std::chrono::steady_clock::now();
	red_t result1 = test(data, size);
	auto end = std::chrono::steady_clock::now();

	std::cout << "CPU (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')'  << result1  << std::endl;

	begin = std::chrono::steady_clock::now();
	red_t result2 = reduction_cuda(data, size, size/ 1024 + 1);
	end = std::chrono::steady_clock::now();

	std::cout << std::endl << "GPU warmup(" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;
	std::cout << "Results are same (" << result2 << ") = " << (result1 == result2 ? "True" : "False") << std::endl;

	begin = std::chrono::steady_clock::now();
	result2 = reduction_cuda(data, size, size / 1024 + 1);
	end = std::chrono::steady_clock::now();

	std::cout << std::endl << "GPU (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;
	std::cout << "Results are same (" << result2 << ") = " << (result1 == result2 ? "True" : "False") << std::endl;


	delete[]data;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
