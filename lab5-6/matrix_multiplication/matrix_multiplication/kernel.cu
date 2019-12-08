#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <cassert>
#include <ostream>
#include <iostream>
#include <random>
#include <chrono>

#define CUBLAS_MAT_MULTIPLY 0

__global__ void multiplyKernel(float* res_mat, float* mat1, float* mat2, size_t m, size_t n, size_t p)
{
	size_t m_ = blockIdx.x * 32 + threadIdx.x;
	size_t p_ = blockIdx.y * 32 + threadIdx.y;


	if (m_ >= m || p_ >= p)
		return;

	float* pos_m1 = mat1 + m_ * n;
	float* pos_m2 = mat2 + p_;

	float* val = res_mat + m_ * p + p_;

	float res = 0;
	for (size_t i = 0; i < n; ++i)
	{
		res += (*(pos_m1 + i)) * (*(pos_m2 + i * p));
	}
	(*val) = res;
}

__global__ void multiplyKernelShared(float* res_mat, float* mat1, float* mat2, size_t m, size_t n, size_t p)
{
	size_t m_ = threadIdx.x;
	size_t p_ = threadIdx.y;
	size_t id = m_ * p + p_;

	extern __shared__ float shared_memory[];
	float* shared_mat_1 = shared_memory;
	float* shared_mat_2 = shared_memory + m * n;

	float* pos_m1 = shared_mat_1 + m_ * n;
	float* pos_m2 = shared_mat_2 + p_;

	if(m == n && n == p)
	{
		shared_mat_1[id] = mat1[id];
		shared_mat_2[id] = mat2[id];
	}
	else
	{
		if(id == 0)
		{
			size_t size = m * n;
			for (unsigned i = 0; i < size; ++i)
			{
				shared_mat_1[i] = mat1[i];
			}
		}
		else if(id == 1)
		{
			size_t size = m * n;

			for (unsigned i = 0; i < size; ++i)
			{
				shared_mat_2[i] = mat2[i];
			}
		}
	}
	__syncthreads();

	float res = 0;
	for (size_t i = 0; i < n; ++i)
	{
		res += (*(pos_m1 + i)) * (*(pos_m2 + i * p));
	}

	//Result
	*(res_mat + m_ * p + p_) = res;
}

class matrix
{
	size_t size_;
	float* data_;

	size_t rows_;
	size_t cols_;

public:
	
	class row
	{
		size_t cols_;
		float* data_;

	public:
		row(float* data, size_t cols) : data_(data), cols_(cols) {}

		float& operator[](size_t col)
		{
			assert(col < cols_);
			return data_[col];
		}

		float operator[](size_t col) const
		{
			assert(col < cols_);
			return data_[col];
		}

		size_t cols()
		{
			return cols_;
		}

	};
	
	matrix(unsigned rows, unsigned cols, float value = 0.) : size_(rows* cols), data_(new float[size_]), rows_(rows), cols_(cols)
	{
		std::fill(data_, data_ + rows_ * cols_, value);
	}
	matrix(matrix& other) : size_(other.cols_* other.rows_), data_(new float[size_]), rows_(other.rows_), cols_(other.cols_)
	{
		for(unsigned i = 0; i < size_; ++i)
		{
			data_[i] = other.data_[i];
		}
	}
	matrix(matrix&& other) noexcept : size_(other.size_), data_(other.data_), rows_(other.rows_), cols_(other.cols_)
	{
		other.data_ = nullptr;
	}	
	~matrix()
	{
		delete data_;
	}

	float* data()
	{
		return data_;
	}

	float* begin()
	{
		return data_;
	}

	float* end()
	{
		return data_ + size_;
	}

	size_t size()
	{
		return size_;
	}

	size_t rows() const
	{
		return rows_;
	}
	
	size_t cols() const
	{
		return cols_;
	}
	
	row operator[](size_t row) const
	{
		assert(row < rows_);
		return matrix::row(data_ + row * cols_, cols_);
	}

	matrix multiply(const matrix& other) const
	{
		matrix result(rows_, other.cols_);

		for(size_t m = 0; m < rows_; ++m)
		{
			for (size_t p = 0; p < other.cols_; ++p)
			{
				float& val = result[m][p];
				
				for (size_t n = 0; n < cols_; ++n)
				{
					val += this->operator[](m)[n] * other[n][p];
				}
			}
		}

		return result;
	}

	matrix operator *(const matrix& other) const
	{
		return multiply(other);
	}

	bool operator ==(matrix& other)
	{
		if(rows_ != other.rows() || cols_ != other.cols())
		{
			return false;
		}

		const float epsilon = 1.;

		for(auto i = 0; i < size_; ++i)
		{
			if(abs(data_[i] - other.data_[i]) > epsilon)
			{
				return false;
			}
		}

		return true;
	}

	matrix& transpose()
	{
		float* tmp_storage = new float[size_];

		for(auto i = 0; i < rows_; ++i)
		{
			for (auto j = 0; j < cols_; ++j)
			{
				tmp_storage[j * cols_ + i] = data_[i * cols_ + j];
			}
		}

		return *this;
	}
};

std::ostream& operator<<(std::ostream& stream, matrix::row row)
{
	stream << '[';

	for(unsigned i = 0; i <	row.cols(); ++i)
	{
		stream << row[i] << (i == row.cols() - 1 ? "" : ", ");
	}
	stream << ']';
	return stream;
}

std::ostream& operator<<(std::ostream& stream, matrix& mat)
{
	stream << '[';

	for(unsigned i = 0; i < mat.rows(); i++)
	{
		stream << mat[i] << (i == mat.rows() - 1 ? "" : "\n");
	}
	stream << ']';
	return stream;
}

matrix multiplyWithCuda(matrix& m1, matrix& m2)
{
	matrix res(m1.rows(), m2.cols());

	float* dev_mat1 = 0;
	float* dev_mat2 = 0;
	float* dev_mat_res = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_mat1, m1.size() * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_mat1, m1.data(), m1.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mat2, m2.size() * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyAsync(dev_mat2, m2.data(), m2.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mat_res, res.size() * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threads_per_block(m1.rows() < 32 ? m1.rows() : 32, m2.cols() < 32 ? m2.cols() : 32);
	dim3 blocks_per_grid(m1.rows() / 32 + 1, m2.cols() / 32 + 1);
	multiplyKernel <<<blocks_per_grid, threads_per_block >>>(dev_mat_res, dev_mat1, dev_mat2, m1.rows(), m1.cols(), m2.cols());

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
	cudaStatus = cudaMemcpy(res.data(), dev_mat_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_mat_res);
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);

	return res;
}

matrix multiplyWithCudaShared(matrix& m1, matrix& m2)
{
	matrix res(m1.rows(), m2.cols());

	if (m1.rows() * m2.cols() > 1024)
	{
		std::cout << "ERROR: Matrix too large";
		return res;
	}


	float* dev_mat1 = 0;
	float* dev_mat2 = 0;
	float* dev_mat_res = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_mat1, m1.size() * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_mat1, m1.data(), m1.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mat2, m2.size() * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyAsync(dev_mat2, m2.data(), m2.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mat_res, res.size() * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threads_per_block(m1.rows() < 32 ? m1.rows() : 32, m2.cols() < 32 ? m2.cols() : 32);
	dim3 blocks_per_grid(m1.rows() / 32 + 1, m2.cols() / 32 + 1);

	size_t shared_size = m1.size() * sizeof(float) + m2.size() * sizeof(float);
	//multiplyKernel << <blocks_per_grid, threads_per_block >> > (dev_mat_res, dev_mat1, dev_mat2, m1.rows(), m1.cols(), m2.cols());
	multiplyKernelShared<<<blocks_per_grid, threads_per_block, shared_size >>>(dev_mat_res, dev_mat1, dev_mat2, m1.rows(), m1.cols(), m2.cols());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "multiplyKernelShared launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multiplyKernelShared!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(res.data(), dev_mat_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_mat_res);
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);

	return res;
}

#if CUBLAS_MAT_MULTIPLY

matrix multiplyWithCuBlas(matrix& m1, matrix& m2)
{
	matrix res(m2.cols(), m1.rows());
	cudaError_t cudaStatus;
	// Allocate 3 arrays on GPU
	// Swap matrices because of the column major indexing
	float* d_mat1, * d_mat2, * d_mat_res;
	cudaMalloc(&d_mat2, m2.rows() * m2.cols() * sizeof(float));
	cudaMalloc(&d_mat1, m1.rows() * m1.cols() * sizeof(float));
	cudaMalloc(&d_mat_res, res.rows() * res.cols() * sizeof(float));

	// Optionally we can copy the data back on CPU and print the arrays
	cudaStatus = cudaMemcpy(m2.data(), d_mat1, m1.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(m1.data(), d_mat2, m2.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int lda = m2.rows(), ldb = m2.cols(), ldc = m1.cols();
	const float alf = 1;
	const float bet = 0;
	const float* alpha = &alf;
	const float* beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m2.rows(), m2.cols(), m1.cols(), alpha, d_mat1, lda, d_mat2, ldb, beta, d_mat_res, ldc);
	cublasDestroy(handle);

	// Copy (and print) the result on host memory
	cudaStatus = cudaMemcpy(res.data(), d_mat_res, res.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	Error:
	//Free GPU memory
	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_mat_res);

	// Transpose because of the column major indexing
	return res.transpose();
}

#endif

int main()
{
	std::default_random_engine engine;
	std::uniform_int_distribution<int> dist(0, 4);

#if 1

	for(unsigned size = 16; size < 1024; ++size)
	{
		std::cout << size << ' ';
		
		matrix mat1(size, size);
		matrix mat2(size, size);

		for (auto& i : mat1)
		{
			i = dist(engine);
		}

		for (auto& i : mat2)
		{
			i = dist(engine);
		}

		auto begin = std::chrono::steady_clock::now();
		matrix mat = mat1 * mat2;
		auto end = std::chrono::steady_clock::now();

		std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ' ';

		if(size < 32)
		{
			begin = std::chrono::steady_clock::now();
			matrix mc2 = multiplyWithCudaShared(mat1, mat2);
			end = std::chrono::steady_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ' ';
		}

		begin = std::chrono::steady_clock::now();
		matrix m3 = multiplyWithCuda(mat1, mat2);
		end = std::chrono::steady_clock::now();

		std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
	}

#else

	matrix mat1(3, 3);
	matrix mat2(3, 3);

	for (auto& i : mat1)
	{
		i = dist(engine);
	}

	for (auto& i : mat2)
	{
		i = dist(engine);
	}

	auto begin = std::chrono::steady_clock::now();
	matrix mat = mat1 * mat2;
	auto end = std::chrono::steady_clock::now();

	std::cout << "Normal matrix multiplication (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;

	begin = std::chrono::steady_clock::now();
	matrix mc = multiplyWithCuda(mat1, mat2);
	end = std::chrono::steady_clock::now();

	std::cout << std::endl << "CUDA matrix multiplication warmup (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;
	std::cout << "Results are same = " << (mat == mc ? "True" : "False") << std::endl;

	begin = std::chrono::steady_clock::now();
	matrix mc2 = multiplyWithCudaShared(mat1, mat2);
	end = std::chrono::steady_clock::now();
	std::cout << std::endl << "CUDA matrix multiplication using shared memory(" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;
	std::cout << "Results are same = " << (mat == mc2 ? "True" : "False") << std::endl;

	begin = std::chrono::steady_clock::now();
	matrix m3 = multiplyWithCuda(mat1, mat2);
	end = std::chrono::steady_clock::now();

	std::cout << std::endl << "CUDA matrix multiplication real (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;
	std::cout << "Results are same = " << (mat == mc ? "True" : "False") << std::endl;

#if CUBLAS_MAT_MULTIPLY
	begin = std::chrono::steady_clock::now();
	matrix m4 = multiplyWithCuBlas(mat1, mat2);
	end = std::chrono::steady_clock::now();

	std::cout << std::endl << "CUDA matrix multiplication2 (" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << ')' << std::endl;
	std::cout << "Results are same = " << (mat == mc ? "True" : "False") << std::endl;
#endif

#endif
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
