
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <ostream>
#include <iostream>
#include <random>
#include <chrono>


__global__ void multiplyKernel(double* res_mat, double* mat1, double* mat2, size_t m, size_t n, size_t p)
{
	size_t m_ = blockIdx.x * 32 + threadIdx.x;
	size_t p_ = blockIdx.y * 32 + threadIdx.y;

	if (m_ >= m || p_ >= p)
		return;

	double* pos_m1 = mat1 + m_ * n;
	double* pos_m2 = mat2 + p_;

	double* val = res_mat + m_ * p + p_;

	for (size_t i = 0; i < n; ++i)
	{
		(*val) += (*(pos_m1 + i)) * (*(pos_m2 + i * p));
	}
}

class matrix
{
	size_t size_;
	double* data_;

	size_t rows_;
	size_t cols_;

public:
	
	class row
	{
		size_t cols_;
		double* data_;

	public:
		row(double* data, size_t cols) : data_(data), cols_(cols) {}

		double& operator[](size_t col)
		{
			assert(col < cols_);
			return data_[col];
		}

		double operator[](size_t col) const
		{
			assert(col < cols_);
			return data_[col];
		}

		size_t cols()
		{
			return cols_;
		}

	};
	
	matrix(unsigned rows, unsigned cols, double value = 0.) : size_(rows* cols), data_(new double[size_]), rows_(rows), cols_(cols)
	{
		std::fill(data_, data_ + rows_ * cols_, value);
	}
	matrix(matrix& other) : size_(other.cols_* other.rows_), data_(new double[size_]), rows_(other.rows_), cols_(other.cols_)
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

	double* data()
	{
		return data_;
	}

	double* begin()
	{
		return data_;
	}

	double* end()
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
				double& val = result[m][p];
				
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

		const double epsilon = 1.;

		for(auto i = 0; i < size_; ++i)
		{
			if(abs(data_[i] - other.data_[i]) > epsilon)
			{
				return false;
			}
		}

		return true;
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

	double* dev_mat1 = 0;
	double* dev_mat2 = 0;
	double* dev_mat_res = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_mat1, m1.size() * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_mat1, m1.data(), m1.size() * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mat2, m2.size() * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyAsync(dev_mat2, m2.data(), m2.size() * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_mat_res, res.size() * sizeof(double));
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
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(res.data(), dev_mat_res, res.size() * sizeof(double), cudaMemcpyDeviceToHost);
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

int main()
{

	std::default_random_engine engine;
	std::uniform_int_distribution<int> dist(0, 4);
	
	matrix mat1(1000, 1000);
	matrix mat2(1000, 1000);

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

	std::cout << "Normal matrix multiplication (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << ')' << std::endl;

	begin = std::chrono::steady_clock::now();
	matrix mc = multiplyWithCuda(mat1, mat2);
	end = std::chrono::steady_clock::now();

	std::cout << std::endl << "CUDA matrix multiplication (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << ')' << std::endl;

	std::cout << "Results are same = " << (mat == mc ? "True" : "False") << std::endl;

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
