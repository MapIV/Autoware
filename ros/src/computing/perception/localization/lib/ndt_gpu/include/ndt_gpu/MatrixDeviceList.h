#ifndef MAT_DEV_LIST_H_
#define MAT_DEV_LIST_H_

#include <cuda.h>
#include "debug.h"
#include <iostream>
#include <memory>

namespace gpu {
template <typename Scalar, int Rows, int Cols>
class MatrixDeviceList {
public:
	MatrixDeviceList();

	MatrixDeviceList(int mat_num);

	MatrixDeviceList(const MatrixDeviceList<Scalar, Rows, Cols> &other);

	MatrixDeviceList<Scalar, Rows, Cols> &operator=(const MatrixDeviceList<Scalar, Rows, Cols> &other);

	MatrixDeviceList(MatrixDeviceList<Scalar, Rows, Cols> &&other);

	MatrixDeviceList<Scalar, Rows, Cols> &operator=(MatrixDeviceList<Scalar, Rows, Cols> &&other);

	bool copy_from(const MatrixDeviceList<Scalar, Rows, Cols> &other);

	// Return the address of the first element at (row, col)
	CUDAH Scalar *operator()(int row, int col);

	// Return the reference to the element at (mat_id, row, col)
	CUDAH Scalar& operator()(int mat_id, int row, int col);

	CUDAH MatrixDevice<Scalar, Rows, Cols> operator()(int mat_id);

	int size() {
		return mat_num_;
	}

	int rows() {
		return Rows;
	}

	int cols() {
		return Cols;
	}

	void free();
private:
	Scalar *buffer_;
	int mat_num_;
	bool is_copied_;
};

template <typename Scalar, int Rows, int Cols>
MatrixDeviceList<Scalar, Rows, Cols>::MatrixDeviceList()
{
	buffer_ = NULL;
	mat_num_ = 0;
	is_copied_ = false;
}

template <typename Scalar, int Rows, int Cols>
MatrixDeviceList<Scalar, Rows, Cols>::MatrixDeviceList(int mat_num)
{
	buffer_ = NULL;
	mat_num_ = mat_num;
	is_copied_ = false;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(Scalar) * Rows * Cols * mat_num));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(Scalar) * Rows * Cols * mat_num));
	checkCudaErrors(cudaDeviceSynchronize());
}

template <typename Scalar, int Rows, int Cols>
MatrixDeviceList<Scalar, Rows, Cols>::MatrixDeviceList(const MatrixDeviceList<Scalar, Rows, Cols> &other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = true;
}

template <typename Scalar, int Rows, int Cols>
MatrixDeviceList<Scalar, Rows, Cols> &MatrixDeviceList<Scalar, Rows, Cols>::operator=(const MatrixDeviceList<Scalar, Rows, Cols> &other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = true;

	return *this;
}


template <typename Scalar, int Rows, int Cols>
MatrixDeviceList<Scalar, Rows, Cols>::MatrixDeviceList(MatrixDeviceList<Scalar, Rows, Cols> &&other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = false;

	other.buffer_ = NULL;
	other.mat_num_ = 0;
	other.is_copied_ = true;
}

template <typename Scalar, int Rows, int Cols>
MatrixDeviceList<Scalar, Rows, Cols> &MatrixDeviceList<Scalar, Rows, Cols>::operator=(MatrixDeviceList<Scalar, Rows, Cols> &&other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = false;

	other.buffer_ = NULL;
	other.mat_num_ = 0;
	other.is_copied_ = true;

	return *this;
}

template <typename Scalar, int Rows, int Cols>
bool MatrixDeviceList<Scalar, Rows, Cols>::copy_from(const MatrixDeviceList<Scalar, Rows, Cols> &other)
{
	if (mat_num_ != other.mat_num_)
		return false;

	checkCudaErrors(cudaMemcpy(buffer_, other.buffer_, sizeof(Scalar) * Rows * Cols * mat_num_, cudaMemcpyDeviceToDevice));

	return true;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar *MatrixDeviceList<Scalar, Rows, Cols>::operator()(int row, int col)
{
	if (row >= Rows || col >= Cols || row < 0 || col < 0)
		return NULL;

	if (mat_num_ == 0 || buffer_ == NULL)
		return NULL;

	return (buffer_ + row * Cols + col);
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar &MatrixDeviceList<Scalar, Rows, Cols>::operator()(int mat_id, int row, int col)
{
	return buffer_[mat_id + (row * Cols + col) * mat_num_];
}

template <typename Scalar, int Rows, int Cols>
CUDAH MatrixDevice<Scalar, Rows, Cols> MatrixDeviceList<Scalar, Rows, Cols>::operator()(int mat_id)
{
	return MatrixDevice<Scalar, Rows, Cols>(mat_num_, buffer_ + mat_id);
}

template <typename Scalar, int Rows, int Cols>
void MatrixDeviceList<Scalar, Rows, Cols>::free()
{
	if (buffer_ != NULL && !is_copied_) {
		checkCudaErrors(cudaFree(buffer_));
	}

	mat_num_ = 0;
}

}

#endif


