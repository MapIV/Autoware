#include "ndt_gpu/MatrixHost.h"
#include "ndt_gpu/debug.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace gpu {

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::MatrixHost()
{
	if (Rows > 0 && Cols > 0) {
		buffer_ = (Scalar *)malloc(sizeof(Scalar) * Rows * Cols);
		offset_ = 1;
		fr_ = true;
	} else {
		fr_ = false;
		buffer_ = NULL;
		offset_ = 0;
	}
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::MatrixHost(int offset, Scalar *buffer) :
Matrix<Scalar, Rows, Cols>(offset, buffer)
{
	fr_ = false;
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::MatrixHost(const MatrixHost<Scalar, Rows, Cols>& other) {
	if (Rows > 0 && Cols > 0) {
		offset_ = 1;
		fr_ = true;

		buffer_ = (Scalar*)malloc(sizeof(Scalar) * Rows * Cols);

		for (int i = 0; i < Rows; i++) {
			for (int j = 0; j < Cols; j++) {
				buffer_[i * Cols + j] = other.at(i, j);
			}
		}
	}
}

template <typename Scalar, int Rows, int Cols>
__global__ void copyMatrixDevToDev(MatrixDevice<Scalar, Rows, Cols> input, MatrixDevice<Scalar, Rows, Cols> output) {
	int row = threadIdx.x;
	int col = threadIdx.y;

	if (row < Rows && col < Cols)
		output(row, col) = input(row, col);
}

template <typename Scalar, int Rows, int Cols>
bool MatrixHost<Scalar, Rows, Cols>::moveToGpu(MatrixDevice<Scalar, Rows, Cols> output) {
	Scalar *tmp;

	checkCudaErrors(cudaMalloc(&tmp, sizeof(Scalar) * Rows * Cols * offset_));
	checkCudaErrors(cudaMemcpy(tmp, buffer_, sizeof(Scalar) * Rows * Cols * offset_, cudaMemcpyHostToDevice));

	MatrixDevice<Scalar, Rows, Cols> tmp_output(offset_, tmp);

	dim3 block_x(Rows, Cols, 1);
	dim3 grid_x(1, 1, 1);

	copyMatrixDevToDev<Scalar, Rows, Cols><<<grid_x, block_x>>>(tmp_output, output);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(tmp));

	return true;
}

template <typename Scalar, int Rows, int Cols>
bool MatrixHost<Scalar, Rows, Cols>::moveToHost(const MatrixDevice<Scalar, Rows, Cols> input) {
	Scalar *tmp;

	checkCudaErrors(cudaMalloc(&tmp, sizeof(Scalar) * Rows * Cols * offset_));

	MatrixDevice<Scalar, Rows, Cols> tmp_output(offset_, tmp);

	dim3 block_x(Rows, Cols, 1);
	dim3 grid_x(1, 1, 1);

	copyMatrixDevToDev<Scalar, Rows, Cols><<<grid_x, block_x>>>(input, tmp_output);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(buffer_, tmp, sizeof(Scalar) * Rows * Cols * offset_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(tmp));

	return true;

}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols> &MatrixHost<Scalar, Rows, Cols>::operator=(const MatrixHost<Scalar, Rows, Cols> &other)
{
	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			buffer_[(i * Cols + j) * offset_] = other.at(i, j);
		}
	}

	return *this;
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols> &MatrixHost<Scalar, Rows, Cols>::operator=(MatrixHost<Scalar, Rows, Cols> &&other)
{
	if (fr_ && buffer_ != NULL) {
		free(buffer_);
		fr_ = false;
	}

	offset_ = other.offset_;
	fr_ = other.fr_;
	buffer_ = other.buffer_;

	other.offset_ = 0;
	other.fr_ = false;
	other.buffer_ = NULL;

	return *this;
}


template <typename Scalar, int Rows, int Cols>
void MatrixHost<Scalar, Rows, Cols>::debug()
{
	std::cout << *this;
}

template <typename Scalar, int Rows, int Cols>
std::ostream &operator<<(std::ostream &os, const MatrixHost<Scalar, Rows, Cols> &value)
{
	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			os << value.at(i, j) << " ";
		}

		os << std::endl;
	}

	os << std::endl;

	return os;
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::~MatrixHost()
{
	if (fr_ && buffer_ != NULL)
		free(buffer_);
}

template class MatrixHost<float, 3, 3>;
template class MatrixHost<double, 3, 3>;
template class MatrixHost<double, 6, 1>;
template class MatrixHost<double, 3, 6>;		// Point gradient class
template class MatrixHost<double, 18, 6>;		// Point hessian class
template class MatrixHost<double, 24, 1>;
template class MatrixHost<double, 45, 1>;

}
