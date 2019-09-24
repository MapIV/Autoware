#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"
#include <iostream>

namespace gpu {
template <typename Scalar = float, int Rows = 0, int Cols = 0>
class MatrixDevice : public Matrix<Scalar, Rows, Cols> {
public:
	MatrixDevice();

	CUDAH MatrixDevice(int offset, Scalar *buffer);

	CUDAH MatrixDevice(const MatrixDevice<Scalar, Rows, Cols> &other);

	CUDAH bool isEmpty();

	CUDAH MatrixDevice<Scalar, Rows, 1> col(int index);

	CUDAH MatrixDevice<Scalar, Cols, 1> row(int index);

	template <int RSize>
	CUDAH MatrixDevice<Scalar, RSize, 1> col(int row, int col);

	// Extract a row of CSize elements from (row, col)
	template <int CSize>
	CUDAH MatrixDevice<Scalar, 1, CSize> row(int row, int col);

	CUDAH MatrixDevice<Scalar, Rows, Cols>& operator=(const MatrixDevice<Scalar, Rows, Cols> &other);

	MatrixDevice<Scalar, Rows, Cols>& operator=(MatrixDevice<Scalar, Rows, Cols> &&other);

	CUDAH void setBuffer(Scalar *buffer);

	void memFree();
private:
	using Matrix<Scalar, Rows, Cols>::buffer_;
	using Matrix<Scalar, Rows, Cols>::offset_;
	using Matrix<Scalar, Rows, Cols>::fr_;
};

template <typename Scalar, int Rows, int Cols>
CUDAH MatrixDevice<Scalar, Rows, Cols>::MatrixDevice(int offset, Scalar *buffer) :
Matrix<Scalar, Rows, Cols>(offset, buffer){}

template <typename Scalar, int Rows, int Cols>
CUDAH bool MatrixDevice<Scalar, Rows, Cols>::isEmpty()
{
	return (Rows == 0 || Cols == 0 || buffer_ == NULL);
}

template <typename Scalar, int Rows, int Cols>
CUDAH MatrixDevice<Scalar, Rows, Cols>::MatrixDevice(const MatrixDevice<Scalar, Rows, Cols> &other)
{
	buffer_ = other.buffer_;
	offset_ = other.offset_;
	fr_ = false;
}

template <typename Scalar, int Rows, int Cols>
CUDAH MatrixDevice<Scalar, Rows, 1> MatrixDevice<Scalar, Rows, Cols>::col(int index)
{
	return MatrixDevice<Scalar, Rows, 1>(offset_ * Cols, buffer_ + index * offset_);
}

template <typename Scalar, int Rows, int Cols>
CUDAH MatrixDevice<Scalar, Cols, 1> MatrixDevice<Scalar, Rows, Cols>::row(int index)
{
	return MatrixDevice<Scalar, Cols, 1>(offset_, buffer_ + index * Cols * offset_);
}

template <typename Scalar, int Rows, int Cols>
template <int RSize>
CUDAH MatrixDevice<Scalar, RSize, 1> MatrixDevice<Scalar, Rows, Cols>::col(int row, int col)
{
	return MatrixDevice<Scalar, RSize, 1>(offset_ * Cols, buffer_ + (row * Cols + col) * offset_);
}

template <typename Scalar, int Rows, int Cols>
template <int CSize>
CUDAH MatrixDevice<Scalar, 1, CSize> MatrixDevice<Scalar, Rows, Cols>::row(int row, int col)
{
	return MatrixDevice<Scalar, 1, CSize>(offset_, buffer_ + (row * Cols + col) * offset_);
}

template <typename Scalar, int Rows, int Cols>
CUDAH void MatrixDevice<Scalar, Rows, Cols>::setBuffer(Scalar *buffer)
{
	buffer_ = buffer;
}

template <typename Scalar, int Rows, int Cols>
CUDAH MatrixDevice<Scalar, Rows, Cols>& MatrixDevice<Scalar, Rows, Cols>::operator=(const MatrixDevice<Scalar, Rows, Cols> &other)
{

#pragma unroll
	for (int i = 0; i < Rows; i++) {
#pragma unroll
		for (int j = 0; j < Cols; j++) {
			buffer_[(i * Cols + j) * offset_] = other.buffer_[(i * Cols + j) * other.offset_];
		}
	}

	return *this;
}


template <typename Scalar, int Size>
class SquareMatrixDevice : public MatrixDevice<Scalar, Size, Size> {
public:
	SquareMatrixDevice() {}
};


}

#endif
