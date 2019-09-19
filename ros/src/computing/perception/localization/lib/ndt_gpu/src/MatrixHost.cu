#include "ndt_gpu/MatrixHost.h"
#include "ndt_gpu/debug.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace gpu {

template <typename T>
MatrixHost<T>::MatrixHost()
{
	fr_ = false;
}

template <typename T>
MatrixHost<T>::MatrixHost(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	offset_ = 1;

	buffer_ = (T*)malloc(sizeof(T) * rows_ * cols_ * offset_);
	memset(buffer_, 0, sizeof(T) * rows_ * cols_ * offset_);
	fr_ = true;
}

template <typename T>
MatrixHost<T>::MatrixHost(int rows, int cols, int offset, const T *buffer)
{
	rows_ = rows;
	cols_ = cols;
	offset_ = offset;
	buffer_ = buffer;
	fr_ = false;
}

template <typename T>
MatrixHost<T>::MatrixHost(const MatrixHost<T>& other) {
	rows_ = other.rows_;
	cols_ = other.cols_;
	offset_ = other.offset_;
	fr_ = other.fr_;

	if (fr_) {
		buffer_ = (T*)malloc(sizeof(T) * rows_ * cols_ * offset_);
		memcpy(buffer_, other.buffer_, sizeof(T) * rows_ * cols_ * offset_);
	} else {
		buffer_ = other.buffer_;
	}
}

template <typename T>
__global__ void copyMatrixDevToDev(MatrixDevice<T> input, MatrixDevice<T> output) {
	int row = threadIdx.x;
	int col = threadIdx.y;
	int rows_num = input.rows();
	int cols_num = input.cols();

	if (row < rows_num && col < cols_num)
		output(row, col) = input(row, col);
}

template <typename T>
bool MatrixHost<T>::moveToGpu(MatrixDevice<T> output) {
	if (rows_ != output.rows() || cols_ != output.cols())
		return false;

	if (offset_ == output.offset()) {
		checkCudaErrors(cudaMemcpy(output.buffer(), buffer_, sizeof(T) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));
		return true;
	}
	else {
		double *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(T) * rows_ * cols_ * offset_));
		checkCudaErrors(cudaMemcpy(tmp, buffer_, sizeof(T) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));

		MatrixDevice<T> tmp_output(rows_, cols_, offset_, tmp);

		dim3 block_x(rows_, cols_, 1);
		dim3 grid_x(1, 1, 1);

		copyMatrixDevToDev<T><<<grid_x, block_x>>>(tmp_output, output);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(tmp));

		return true;
	}
}

template <typename T>
bool MatrixHost<T>::moveToHost(const MatrixDevice<T> input) {
	if (rows_ != input.rows() || cols_ != input.cols())
		return false;

	if (offset_ == input.offset()) {
		checkCudaErrors(cudaMemcpy(buffer_, input.buffer(), sizeof(T) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
		return true;
	}
	else {
		double *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(T) * rows_ * cols_ * offset_));

		MatrixDevice<T> tmp_output(rows_, cols_, offset_, tmp);

		dim3 block_x(rows_, cols_, 1);
		dim3 grid_x(1, 1, 1);

		copyMatrixDevToDev<T><<<grid_x, block_x>>>(input, tmp_output);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(buffer_, tmp, sizeof(T) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(tmp));

		return true;
	}
}

template <typename T>
MatrixHost<T> &MatrixHost<T>::operator=(const MatrixHost<T> &other)
{
	rows_ = other.rows_;
	cols_ = other.cols_;
	offset_ = other.offset_;
	fr_ = other.fr_;

	if (fr_) {
		buffer_ = (T*)malloc(sizeof(T) * rows_ * cols_ * offset_);
		memcpy(buffer_, other.buffer_, sizeof(T) * rows_ * cols_ * offset_);
	} else {
		buffer_ = other.buffer_;
	}

	return *this;
}

template <typename T>
void MatrixHost<T>::debug()
{
	std::cout << *this;
}

template <typename T>
friend std::ostream &MatrixHost<T>::operator<<(std::ostream &os, const MatrixHost<T> &value)
{
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			os << buffer_[(i * cols_ + j) * offset_] << " ";
		}

		os << std::endl;
	}

	os << std::endl;

	return os;
}

template <typename T>
MatrixHost<T>::~MatrixHost()
{
	if (fr_)
		free(buffer_);
}

}
