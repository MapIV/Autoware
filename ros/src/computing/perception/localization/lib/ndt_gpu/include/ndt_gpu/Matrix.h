#ifndef GMATRIX_H_
#define GMATRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <float.h>

namespace gpu {

template <typename T>
class Matrix {
public:
	CUDAH Matrix();

	CUDAH Matrix(int rows, int cols, int offset, const T *buffer);

	CUDAH Matrix(const Matrix<T> &other);
	CUDAH Matrix(Matrix<T> &&other);

	CUDAH int rows() const;
	CUDAH int cols() const;
	CUDAH int offset() const;

	CUDAH T *buffer() const;

	CUDAH void setRows(int rows);
	CUDAH void setCols(int cols);
	CUDAH void setOffset(int offset);
	CUDAH void setBuffer(const T *buffer);
	CUDAH void setCellVal(int row, int col, T val);

	CUDAH void copy(Matrix<T> &output);

	//Need to fix. Only reducing rows is OK now.
	CUDAH void resize(int rows, int cols);

	CUDAH T *cellAddr(int row, int col);

	CUDAH T *cellAddr(int index);

	//Assignment operator
	// Copy assignment
	CUDAH Matrix<T>& operator=(const Matrix<T> &input);

	CUDAH T& operator()(int row, int col);

	CUDAH void set(int row, int col, T val);

	CUDAH T& operator()(int index);

	CUDAH T at(int row, int col) const;

	template <typename T2>
	CUDAH bool operator*=(T2 val);

	template <typename T2>
	CUDAH bool operator/=(T2 val);

	CUDAH bool transpose(Matrix<T> &output);

	//Only applicable for 3x3 matrix or below
	CUDAH bool inverse(Matrix<T> &output);

	CUDAH Matrix<T> col(int index);

	CUDAH Matrix<T> row(int index);

protected:
	T *buffer_;
	int rows_, cols_, offset_;
};

template <typename T>
CUDAH Matrix<T>::Matrix() {
	buffer_ = NULL;
	rows_ = cols_ = offset_ = 0;
}

template <typename T>
CUDAH Matrix<T>::Matrix(int rows, int cols, int offset, const T *buffer) {
	rows_ = rows;
	cols_ = cols;
	offset_ = offset;
	buffer_ = buffer;
}

template <typename T>
CUDAH Matrix<T>::Matrix(const Matrix<T> &other) {
	rows_ = other.rows_;
	cols_ = other.cols_;
	offset_ = other.offset_;
	buffer_ = other.buffer_;
}

template <typename T>
CUDAH Matrix<T>::Matrix(Matrix<T> &&other) {
	rows_ = other.rows_;
	cols_ = other.cols_;
	offset_ = other.offset_;
	buffer_ = other.buffer_;

	other.rows_ = other.cols_ = other.offset_ = 0;
	other.buffer_ = NULL;
}

template <typename T>
CUDAH int Matrix<T>::rows() const {
	return rows_;
}

template <typename T>
CUDAH int Matrix<T>::cols() const {
	return cols_;
}

template <typename T>
CUDAH int Matrix<T>::offset() const {
	return offset_;
}

template <typename T>
CUDAH T *Matrix<T>::buffer() const {
	return buffer_;
}


template <typename T> CUDAH void Matrix<T>::setRows(int rows) { rows_ = rows; }
template <typename T> CUDAH void Matrix<T>::setCols(int cols) { cols_ = cols; }
template <typename T> CUDAH void Matrix<T>::setOffset(int offset) { offset_ = offset; }
template <typename T> CUDAH void Matrix<T>::setBuffer(const T *buffer) { buffer_ = buffer; }
template <typename T> CUDAH void Matrix<T>::setCellVal(int row, int col, T val) {
	buffer_[(row * cols_ + col) * offset_] = val;
}

template <typename T>
CUDAH void Matrix<T>::copy(Matrix<T> &output) {
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			output(i, j) = buffer_[(i * cols_ + j) * offset_];
		}
	}
}

//Need to fix. Only reducing rows is OK now.
template <typename T>
CUDAH void Matrix<T>::resize(int rows, int cols) {
	rows_ = rows;
	cols_ = cols;
}

template <typename T>
CUDAH T *Matrix<T>::cellAddr(int row, int col) {
	if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
		return NULL;

	return buffer_ + (row * cols_ + col) * offset_;
}

template <typename T>
CUDAH T *Matrix<T>::cellAddr(int index) {
	if (rows_ == 1 && index >= 0 && index < cols_) {
			return buffer_ + index * offset_;
	}
	else if (cols_ == 1 && index >= 0 && index < rows_) {
			return buffer_ + index * offset_;
	}

	return NULL;
}

//Assignment operator
template <typename T>
CUDAH Matrix<T>& Matrix<T>::operator=(const Matrix<T> &input) {
	rows_ = input.rows_;
	cols_ = input.cols_;
	offset_ = input.offset_;
	buffer_ = input.buffer_;

	return *this;
}

template<typename T>
CUDAH T& Matrix<T>::operator()(int row, int col) {
	return buffer_[(row * cols_ + col) * offset_];
}

template <typename T>
CUDAH void Matrix<T>::set(int row, int col, T val) {
	buffer_[(row * cols_ + col) * offset_] = val;
}

template <typename T>
CUDAH T& Matrix<T>::operator()(int index) {
	return buffer_[index * offset_];
}

template <typename T>
CUDAH T Matrix<T>::at(int row, int col) const {
	return buffer_[(row * cols_ + col) * offset_];
}

template <typename T>
template <typename T2>
CUDAH bool Matrix<T>::operator*=(T2 val) {
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			buffer_[(i * cols_ + j) * offset_] *= val;
		}
	}

	return true;
}

template <typename T>
template <typename T2>
CUDAH bool Matrix<T>::operator/=(T2 val) {
	if (val == 0)
		return false;

	for (int i = 0; i < rows_ * cols_; i++) {
			buffer_[i * offset_] /= val;
	}

	return true;
}

template <typename T>
CUDAH bool Matrix<T>::transpose(Matrix<T> &output) {
	if (rows_ * cols_ != output.rows_ * output.cols_)
		return false;

	output.rows_ = cols_;
	output.cols_ = rows_;

	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			output(j, i) = buffer_[(i * cols_ + j) * offset_];
		}
	}

	return true;
}

//Only applicable for 3x3 matrix or below
template <typename T>
CUDAH bool Matrix<T>::inverse(Matrix<T> &output) {
	if (rows_ != cols_ || rows_ == 0 || cols_ == 0)
		return false;

	if (rows_ == 1) {
		if (buffer_[0] != 0)
			output(0, 0) = 1 / buffer_[0];
		else
			return false;
	}

	if (rows_ == 2) {
		T det = at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);

		if (det != 0) {
			output(0, 0) = at(1, 1) / det;
			output(0, 1) = - at(0, 1) / det;

			output(1, 0) = - at(1, 0) / det;
			output(1, 1) = at(0, 0) / det;
		} else
			return false;
	}

	if (rows_ == 3) {
		T det = at(0, 0) * at(1, 1) * at(2, 2) + at(0, 1) * at(1, 2) * at(2, 0) + at(1, 0) * at (2, 1) * at(0, 2)
						- at(0, 2) * at(1, 1) * at(2, 0) - at(0, 1) * at(1, 0) * at(2, 2) - at(0, 0) * at(1, 2) * at(2, 1);
		T idet = 1.0 / det;

		if (det != 0) {
			output(0, 0) = (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) * idet;
			output(0, 1) = - (at(0, 1) * at(2, 2) - at(0, 2) * at(2, 1)) * idet;
			output(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * idet;

			output(1, 0) = - (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) * idet;
			output(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * idet;
			output(1, 2) = - (at(0, 0) * at(1, 2) - at(0, 2) * at(1, 0)) * idet;

			output(2, 0) = (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0)) * idet;
			output(2, 1) = - (at(0, 0) * at(2, 1) - at(0, 1) * at(2, 0)) * idet;
			output(2, 2) = (at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0)) * idet;
		} else
			return false;
	}

	return true;
}

template <typename T>
CUDAH Matrix<T> Matrix<T>::col(int index) {
	return Matrix<T>(rows_, 1, offset_ * cols_, buffer_ + index * offset_);
}

template <typename T>
CUDAH Matrix<T> Matrix<T>::row(int index) {
	return Matrix<T>(1, cols_, offset_, buffer_ + index * cols_ * offset_);
}

}

#endif
