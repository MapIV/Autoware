#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"

namespace gpu {
template <typename T>
class MatrixDevice : public Matrix<T> {
public:
	CUDAH MatrixDevice();

	CUDAH MatrixDevice(int rows, int cols);

	CUDAH MatrixDevice(int rows, int cols, int offset, const T *buffer);

	CUDAH bool isEmpty();

	CUDAH MatrixDevice<T> col(int index);

	CUDAH MatrixDevice<T> row(int index);

	CUDAH void setBuffer(const T *buffer);

	void memAlloc();

	void memFree();

private:
	bool fr_;
};

template <typename T>
CUDAH MatrixDevice<T>::MatrixDevice()
{
	rows_ = cols_ = offset_ = 0;
	buffer_ = NULL;
	fr_ = true;
}

template <typename T>
CUDAH MatrixDevice<T>::MatrixDevice(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
	offset_ = 1;
	fr_ = true;
	buffer_ = NULL;
}

template <typename T>
CUDAH MatrixDevice<T>::MatrixDevice(int rows, int cols, int offset, const T *buffer)
{
	rows_ = rows;
	cols_ = cols;
	offset_ = offset;
	buffer_ = buffer;
	fr_ = false;
}

template <typename T>
CUDAH bool MatrixDevice<T>::isEmpty()
{
	return (rows_ == 0 || cols_ == 0 || buffer_ == NULL);
}

template <typename T>
CUDAH MatrixDevice<T> MatrixDevice<T>::col(int index)
{
	return MatrixDevice<T>(rows_, 1, offset_ * cols_, buffer_ + index * offset_);
}

template <typename T>
CUDAH MatrixDevice<T> MatrixDevice<T>::row(int index)
{
	return MatrixDevice<T>(1, cols_, offset_, buffer_ + index * cols_ * offset_);
}

template <typename T>
CUDAH void MatrixDevice<T>::setBuffer(const T *buffer)
{
	buffer_ = buffer;
}



template <typename T>
class SquareMatrixDevice : public MatrixDevice<T> {
public:
	SquareMatrixDevice(int size);
};

}

template class MatrixDevice<float>;
template class MatrixDevice<double>;

#endif
