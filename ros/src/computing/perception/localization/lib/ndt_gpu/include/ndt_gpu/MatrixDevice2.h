#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"
#include <iostream>

namespace gpu {
template <typename Scalar = float>
class MatrixDevice2 {
public:
	MatrixDevice2();

	CUDAH MatrixDevice2(int rows, int cols, int offset, Scalar *buffer);

	CUDAH MatrixDevice2(const MatrixDevice2<Scalar> &other);

	CUDAH bool isEmpty();

	CUDAH MatrixDevice2<Scalar> col(int index);

	CUDAH MatrixDevice2<Scalar> row(int index);

	CUDAH MatrixDevice2<Scalar> col(int row, int col, int rsize);

	CUDAH MatrixDevice2<Scalar> row(int row, int col, int csize);

	CUDAH MatrixDevice2<Scalar>& operator=(const MatrixDevice2<Scalar> &other);

	MatrixDevice2<Scalar>& operator=(MatrixDevice2<Scalar> &&other);

	CUDAH void setBuffer(Scalar *buffer);

	void memFree();
private:
	Scalar *buffer_;
	int offset_;
	bool fr_;
	int rows_, cols_;
};

template <typename Scalar>
CUDAH MatrixDevice2<Scalar>::MatrixDevice2(int rows, int cols, int offset, Scalar *buffer)
{
	rows_ = rows;
	cols_ = cols;
	fr_ = false;
	buffer_ = buffer;
	offset_ = offset;
}

template <typename Scalar>
CUDAH bool MatrixDevice2<Scalar>::isEmpty()
{
	return (rows_ == 0 || cols_ == 0 || buffer_ == NULL);
}

template <typename Scalar>
CUDAH MatrixDevice2<Scalar>::MatrixDevice2(const MatrixDevice2<Scalar> &other)
{
	buffer_ = other.buffer_;
	offset_ = other.offset_;
	fr_ = false;
	rows_ = other.rows_;
	cols_ = other.cols_;
}

template <typename Scalar>
CUDAH MatrixDevice2<Scalar> MatrixDevice2<Scalar>::col(int index)
{
	return MatrixDevice2<Scalar>(rows_, 1, offset_ * cols_, buffer_ + index * offset_);
}

template <typename Scalar>
CUDAH MatrixDevice2<Scalar> MatrixDevice2<Scalar>::row(int index)
{
	return MatrixDevice2<Scalar>(1, cols_, offset_, buffer_ + index * cols_ * offset_);
}

template <typename Scalar>
CUDAH MatrixDevice2<Scalar> MatrixDevice2<Scalar>::col(int row, int col, int rsize)
{
	return MatrixDevice2<Scalar>(rsize, 1, offset_ * cols_, buffer_ + (row * cols_ + col) * offset_);
}

template <typename Scalar>
CUDAH MatrixDevice2<Scalar> MatrixDevice2<Scalar>::row(int row, int col, int csize)
{
	return MatrixDevice2<Scalar>(1, csize, offset_, buffer_ + (row * cols_ + col) * offset_);
}

template <typename Scalar>
CUDAH void MatrixDevice2<Scalar>::setBuffer(Scalar *buffer)
{
	buffer_ = buffer;
}

template <typename Scalar>
CUDAH MatrixDevice2<Scalar>& MatrixDevice2<Scalar>::operator=(const MatrixDevice2<Scalar> &other)
{

	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			buffer_[(i * cols_ + j) * offset_] = other.buffer_[(i * cols_ + j) * other.offset_];
		}
	}

	return *this;
}


template <typename Scalar, int Size>
class SquareMatrixDevice2 : public MatrixDevice2<Scalar, Size, Size> {
public:
	SquareMatrixDevice2() {}
};


}

#endif
