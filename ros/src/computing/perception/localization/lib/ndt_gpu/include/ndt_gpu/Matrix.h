#ifndef GMAScalarRIX_H_
#define GMAScalarRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <float.h>

namespace gpu {

template <typename Scalar>
class Matrix {
public:
	CUDAH Matrix();

	CUDAH Matrix(int rows, int cols, int offset, Scalar *buffer);

	CUDAH Matrix(const Matrix<Scalar> &other);
	CUDAH Matrix(Matrix<Scalar> &&other);

	CUDAH int rows() const;
	CUDAH int cols() const;
	CUDAH int offset() const;

	CUDAH Scalar *buffer() const;

	CUDAH void setOffset(int offset);
	CUDAH void setBuffer(Scalar *buffer);
	CUDAH void setCellVal(int row, int col, Scalar val);

	// Deep copy to output
	CUDAH bool copy_from(const Matrix<Scalar> &output);

	//Assignment operator
	// Copy assignment
	CUDAH Matrix<Scalar>& operator=(const Matrix<Scalar> &input);

	CUDAH void set(int row, int col, Scalar val);

	CUDAH Scalar at(int row, int col) const;
	CUDAH Scalar at(int idx) const;

	// Operators
	CUDAH Scalar& operator()(int row, int col);
	CUDAH Scalar& operator()(int index);

	template <typename Scalar2>
	CUDAH Matrix<Scalar>& operator*=(Scalar2 val);

	template <typename Scalar2>
	CUDAH Matrix<Scalar>& operator/=(Scalar2 val);

	CUDAH bool transpose(Matrix<Scalar> &output);

	//Only applicable for 3x3 matrix or below
	CUDAH bool inverse(Matrix<Scalar> &output);

	CUDAH Scalar dot(const Matrix<Scalar> &other);

	CUDAH Matrix<Scalar> col(int index);

	CUDAH Matrix<Scalar> row(int index);

	// Extract a col of RSize elements from (row, col)
	CUDAH Matrix<Scalar> col(int row, int col, int rsize);

	// Extract a row of CSize elements from (row, col)
	CUDAH Matrix<Scalar> row(int row, int col, int csize);

protected:
	Scalar *buffer_;
	int offset_;
	bool is_copied_;	// True: free buffer after being used, false: do nothing
	int rows_, cols_;
};

template <typename Scalar>
CUDAH Matrix<Scalar>::Matrix() {
	offset_ = 0;
	buffer_ = NULL;
	is_copied_ = false;
	rows_ = cols_ = 0;
}

template <typename Scalar>
CUDAH Matrix<Scalar>::Matrix(int rows, int cols, int offset, Scalar *buffer) {
	offset_ = offset;
	buffer_ = buffer;
	is_copied_ = true;
	rows_ = rows;
	cols_ = cols;
}

template <typename Scalar>
CUDAH Matrix<Scalar>::Matrix(const Matrix<Scalar> &other) {
	offset_ = other.offset_;
	buffer_ = other.buffer_;
	is_copied_ = true;
	rows_ = other.rows_;
	cols_ = other.cols_;
}

template <typename Scalar>
CUDAH Matrix<Scalar>::Matrix(Matrix<Scalar> &&other) {
	offset_ = other.offset_;
	buffer_ = other.buffer_;
	is_copied_ = false;
	rows_ = other.rows_;
	cols_ = other.cols_;

	other.offset_ = 0;
	other.buffer_ = NULL;
	other.is_copied_ = true;
	other.rows_ = other.cols_ = 0;
}

template <typename Scalar>
CUDAH int Matrix<Scalar>::rows() const {
	return rows_;
}

template <typename Scalar>
CUDAH int Matrix<Scalar>::cols() const {
	return cols_;
}

template <typename Scalar>
CUDAH int Matrix<Scalar>::offset() const {
	return offset_;
}

template <typename Scalar>
CUDAH Scalar *Matrix<Scalar>::buffer() const {
	return buffer_;
}


template <typename Scalar> CUDAH void Matrix<Scalar>::setOffset(int offset) { offset_ = offset; }
template <typename Scalar> CUDAH void Matrix<Scalar>::setBuffer(Scalar *buffer) { buffer_ = buffer; }
template <typename Scalar> CUDAH void Matrix<Scalar>::setCellVal(int row, int col, Scalar val) {
	buffer_[(row * cols_ + col) * offset_] = val;
}

template <typename Scalar>
CUDAH bool Matrix<Scalar>::copy_from(const Matrix<Scalar> &output) {
	if (rows_ == output.rows_ && cols_ == output.cols_) {
		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				buffer_[(i * cols_ + j) * offset_] = output.at(i, j);
			}
		}

		return true;
	}

	return false;
}

//Copy assignment
template <typename Scalar>
CUDAH Matrix<Scalar>& Matrix<Scalar>::operator=(const Matrix<Scalar> &other) {
	offset_ = other.offset_;
	buffer_ = other.buffer_;
	rows_ = other.rows_;
	cols_ = other.cols_;
	is_copied_ = true;


	return *this;
}

template<typename Scalar>
CUDAH Scalar& Matrix<Scalar>::operator()(int row, int col) {
	return buffer_[(row * cols_ + col) * offset_];
}

template <typename Scalar>
CUDAH void Matrix<Scalar>::set(int row, int col, Scalar val) {
	buffer_[(row * cols_ + col) * offset_] = val;
}

template <typename Scalar>
CUDAH Scalar& Matrix<Scalar>::operator()(int index) {
	return buffer_[index * offset_];
}

template <typename Scalar>
CUDAH Scalar Matrix<Scalar>::at(int row, int col) const {
	return buffer_[(row * cols_ + col) * offset_];
}

template <typename Scalar>
CUDAH Scalar Matrix<Scalar>::at(int idx) const {
	return buffer_[idx * offset_];
}

template <typename Scalar>
template <typename Scalar2>
CUDAH Matrix<Scalar>& Matrix<Scalar>::operator*=(Scalar2 val)
{
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			buffer_[(i * cols_ + j) * offset_] *= val;
		}
	}

	return *this;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH Matrix<Scalar>& Matrix<Scalar>::operator/=(Scalar2 val)
{
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			buffer_[(i * cols_ + j) * offset_] /= val;
		}
	}

	return *this;
}

template <typename Scalar>
CUDAH bool Matrix<Scalar>::transpose(Matrix<Scalar> &output)
{
	if (rows_ == output.cols_ && cols_ == output.rows_) {
		for (int i = 0; i < rows_; i++) {
			for (int j = 0; j < cols_; j++) {
				output(j, i) = buffer_[(i * cols_ + j) * offset_];
			}
		}

		return true;
	}

	return false;
}

//Only applicable for 3x3 matrix or below
template <typename Scalar>
CUDAH bool Matrix<Scalar>::inverse(Matrix<Scalar> &output) {
	return true;
}

template <typename Scalar>
CUDAH Scalar Matrix<Scalar>::dot(const Matrix<Scalar> &other)
{
	Scalar res = 0;

	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			res += buffer_[(i * rows_ + j) * offset_] * other.at(i, j);
		}
	}

	return res;
}

template <typename Scalar>
CUDAH Matrix<Scalar> Matrix<Scalar>::col(int index) {
	return Matrix<Scalar>(rows_, 1, offset_ * cols_, buffer_ + index * offset_);
}

template <typename Scalar>
CUDAH Matrix<Scalar> Matrix<Scalar>::row(int index) {
	return Matrix<Scalar>(1, cols_, offset_, buffer_ + index * cols_ * offset_);
}

template <typename Scalar>
CUDAH Matrix<Scalar> Matrix<Scalar>::col(int row, int col, int rsize)
{
	return Matrix<Scalar>(rsize, 1, offset_ * cols_, buffer_ + (row * cols_ + col) * offset_);
}

template <typename Scalar>
CUDAH Matrix<Scalar> Matrix<Scalar>::row(int row, int col, int csize)
{
	return Matrix<Scalar>(1, csize, offset_, buffer_ + (row * cols_ + col) * offset_);
}

template <>
CUDAH bool Matrix<double>::inverse(Matrix<double> &output) {
	if (rows_ != cols_)
		return false;

	if (rows_ == 3) {
		double det = at(0, 0) * at(1, 1) * at(2, 2) + at(0, 1) * at(1, 2) * at(2, 0) + at(1, 0) * at (2, 1) * at(0, 2)
						- at(0, 2) * at(1, 1) * at(2, 0) - at(0, 1) * at(1, 0) * at(2, 2) - at(0, 0) * at(1, 2) * at(2, 1);

		double idet = 1.0 / det;

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


template class Matrix<float>;
template class Matrix<double>;
}

#endif
