#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"
#include <iostream>

namespace gpu {
template <typename Scalar = float>
class MatrixDevice : public Matrix<Scalar> {
public:
	MatrixDevice();

	MatrixDevice(int rows, int cols);

	CUDAH MatrixDevice(int rows, int cols, int offset, Scalar *buffer);

	CUDAH MatrixDevice(const MatrixDevice<Scalar> &other);

	CUDAH bool isEmpty();

	CUDAH MatrixDevice<Scalar> col(int index);

	CUDAH MatrixDevice<Scalar> row(int index);

	CUDAH MatrixDevice<Scalar> col(int row, int col, int rsize);

	// Extract a row of CSize elements from (row, col)
	CUDAH MatrixDevice<Scalar> row(int row, int col, int csize);

	MatrixDevice<Scalar>& operator=(const MatrixDevice<Scalar> &other);

	MatrixDevice<Scalar>& operator=(MatrixDevice<Scalar> &&other);

	CUDAH bool copy_from(const MatrixDevice<Scalar> &other);

	CUDAH void setBuffer(Scalar *buffer);

	void free();
private:
	using Matrix<Scalar>::buffer_;
	using Matrix<Scalar>::offset_;
	using Matrix<Scalar>::is_copied_;
	using Matrix<Scalar>::rows_;
	using Matrix<Scalar>::cols_;
};

template <typename Scalar>
CUDAH MatrixDevice<Scalar>::MatrixDevice(int rows, int cols, int offset, Scalar *buffer) :
Matrix<Scalar>(rows, cols, offset, buffer){}

template <typename Scalar>
CUDAH bool MatrixDevice<Scalar>::isEmpty()
{
	return (rows_ == 0 || cols_ == 0 || buffer_ == NULL);
}

template <typename Scalar>
CUDAH MatrixDevice<Scalar>::MatrixDevice(const MatrixDevice<Scalar> &other)
{
	buffer_ = other.buffer_;
	offset_ = other.offset_;
	rows_ = other.rows_;
	cols_ = other.cols_;
	is_copied_ = true;
}

template <typename Scalar>
CUDAH MatrixDevice<Scalar> MatrixDevice<Scalar>::col(int index)
{
	return MatrixDevice<Scalar>(rows_, 1, offset_ * cols_, buffer_ + index * offset_);
}

template <typename Scalar>
CUDAH MatrixDevice<Scalar> MatrixDevice<Scalar>::row(int index)
{
	return MatrixDevice<Scalar>(1, cols_, offset_, buffer_ + index * cols_ * offset_);
}

template <typename Scalar>
CUDAH MatrixDevice<Scalar> MatrixDevice<Scalar>::col(int row, int col, int rsize)
{
	return MatrixDevice<Scalar>(rsize, 1, offset_ * cols_, buffer_ + (row * cols_ + col) * offset_);
}

template <typename Scalar>
CUDAH MatrixDevice<Scalar> MatrixDevice<Scalar>::row(int row, int col, int csize)
{
	return MatrixDevice<Scalar>(1, csize, offset_, buffer_ + (row * cols_ + col) * offset_);
}

template <typename Scalar>
CUDAH void MatrixDevice<Scalar>::setBuffer(Scalar *buffer)
{
	buffer_ = buffer;
}

template <typename Scalar>
CUDAH bool MatrixDevice<Scalar>::copy_from(const MatrixDevice<Scalar> &other)
{
	if (rows_ == other.rows_ && cols_ == other.cols_) {
		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				buffer_[(i * cols_ + j) * offset_] = other.buffer_[(i * cols_ + j) * other.offset_];
			}
		}

		return true;
	}

	return false;
}

}

#endif
