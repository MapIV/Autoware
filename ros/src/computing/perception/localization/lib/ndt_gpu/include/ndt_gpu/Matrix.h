#ifndef GMAScalarRIX_H_
#define GMAScalarRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <float.h>

namespace gpu {

template <typename Scalar, int Rows = 0, int Cols = 0>
class Matrix {
public:
	CUDAH Matrix();

	CUDAH Matrix(int offset, Scalar *buffer);

	CUDAH Matrix(const Matrix<Scalar, Rows, Cols> &other);
	CUDAH Matrix(Matrix<Scalar, Rows, Cols> &&other);

	CUDAH int rows() const;
	CUDAH int cols() const;
	CUDAH int offset() const;

	CUDAH Scalar *buffer() const;

	CUDAH void setOffset(int offset);
	CUDAH void setBuffer(Scalar *buffer);
	CUDAH void setCellVal(int row, int col, Scalar val);

	// Deep copy to output
	CUDAH void copy(Matrix<Scalar, Rows, Cols> &output);

	CUDAH Scalar *cellAddr(int row, int col);

	CUDAH Scalar *cellAddr(int index);

	//Assignment operator
	// Copy assignment
	CUDAH Matrix<Scalar, Rows, Cols>& operator=(const Matrix<Scalar, Rows, Cols> &input);

	CUDAH void set(int row, int col, Scalar val);

	CUDAH Scalar at(int row, int col) const;

	// Operators
	CUDAH Scalar& operator()(int row, int col);
	CUDAH Scalar& operator()(int index);

	template <typename Scalar2>
	CUDAH Matrix<Scalar, Rows, Cols>& operator*=(Scalar2 val);

	template <typename Scalar2>
	CUDAH Matrix<Scalar, Rows, Cols>& operator/=(Scalar2 val);

	CUDAH bool transpose(Matrix<Scalar, Cols, Rows> &output);

	//Only applicable for 3x3 matrix or below
	CUDAH bool inverse(Matrix<Scalar, Rows, Cols> &output);

	//CUDAH Scalar dot(const Matrix<Scalar, Rows, Cols> &other);

	CUDAH Matrix<Scalar, Rows, 1> col(int index);

	CUDAH Matrix<Scalar, 1, Cols> row(int index);

	// Extract a col of RSize elements from (row, col)
	template <int RSize>
	CUDAH Matrix<Scalar, RSize, 1> col(int row, int col);

	// Extract a row of CSize elements from (row, col)
	template <int CSize>
	CUDAH Matrix<Scalar, 1, CSize> row(int row, int col);

protected:
	Scalar *buffer_;
	int offset_;
	bool fr_;	// True: free buffer after being used, false: do nothing
};

template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, Rows, Cols>::Matrix() {
	offset_ = 0;
	buffer_ = NULL;
	fr_ = false;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, Rows, Cols>::Matrix(int offset, Scalar *buffer) {
	offset_ = offset;
	buffer_ = buffer;
	fr_ = false;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, Rows, Cols>::Matrix(const Matrix<Scalar, Rows, Cols> &other) {
	offset_ = other.offset_;
	buffer_ = other.buffer_;
	fr_ = false;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, Rows, Cols>::Matrix(Matrix<Scalar, Rows, Cols> &&other) {
	offset_ = other.offset_;
	buffer_ = other.buffer_;
	fr_ = other.fr_;

	other.offset_ = 0;
	other.buffer_ = NULL;
	other.fr_ = false;
}

template <typename Scalar, int Rows, int Cols>
CUDAH int Matrix<Scalar, Rows, Cols>::rows() const {
	return Rows;
}

template <typename Scalar, int Rows, int Cols>
CUDAH int Matrix<Scalar, Rows, Cols>::cols() const {
	return Cols;
}

template <typename Scalar, int Rows, int Cols>
CUDAH int Matrix<Scalar, Rows, Cols>::offset() const {
	return offset_;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar *Matrix<Scalar, Rows, Cols>::buffer() const {
	return buffer_;
}


template <typename Scalar, int Rows, int Cols> CUDAH void Matrix<Scalar, Rows, Cols>::setOffset(int offset) { offset_ = offset; }
template <typename Scalar, int Rows, int Cols> CUDAH void Matrix<Scalar, Rows, Cols>::setBuffer(Scalar *buffer) { buffer_ = buffer; }
template <typename Scalar, int Rows, int Cols> CUDAH void Matrix<Scalar, Rows, Cols>::setCellVal(int row, int col, Scalar val) {
	buffer_[(row * Cols + col) * offset_] = val;
}

template <typename Scalar, int Rows, int Cols>
CUDAH void Matrix<Scalar, Rows, Cols>::copy(Matrix<Scalar, Rows, Cols> &output) {
	for (int i = 0; i < Rows; i++) {
		for (int j = 0; j < Cols; j++) {
			output(i, j) = buffer_[(i * Cols + j) * offset_];
		}
	}
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar *Matrix<Scalar, Rows, Cols>::cellAddr(int row, int col) {
	if (row >= Rows || col >= Cols || row < 0 || col < 0)
		return NULL;

	return buffer_ + (row * Cols + col) * offset_;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar *Matrix<Scalar, Rows, Cols>::cellAddr(int index) {
	if (Rows == 1 && index >= 0 && index < Cols) {
			return buffer_ + index * offset_;
	}
	else if (Cols == 1 && index >= 0 && index < Rows) {
			return buffer_ + index * offset_;
	}

	return NULL;
}

//Assignment operator
template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, Rows, Cols>& Matrix<Scalar, Rows, Cols>::operator=(const Matrix<Scalar, Rows, Cols> &input) {
	offset_ = input.offset_;
	buffer_ = input.buffer_;

	return *this;
}

template<typename Scalar, int Rows, int Cols>
CUDAH Scalar& Matrix<Scalar, Rows, Cols>::operator()(int row, int col) {
	return buffer_[(row * Cols + col) * offset_];
}

template <typename Scalar, int Rows, int Cols>
CUDAH void Matrix<Scalar, Rows, Cols>::set(int row, int col, Scalar val) {
	buffer_[(row * Cols + col) * offset_] = val;
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar& Matrix<Scalar, Rows, Cols>::operator()(int index) {
	return buffer_[index * offset_];
}

template <typename Scalar, int Rows, int Cols>
CUDAH Scalar Matrix<Scalar, Rows, Cols>::at(int row, int col) const {
	return buffer_[(row * Cols + col) * offset_];
}

template <typename Scalar, int Rows, int Cols>
template <typename Scalar2>
CUDAH Matrix<Scalar, Rows, Cols>& Matrix<Scalar, Rows, Cols>::operator*=(Scalar2 val)
{
#pragma unroll
	for (int i = 0; i < Rows; i++) {
#pragma unroll
		for (int j = 0; j < Cols; j++) {
			buffer_[(i * Cols + j) * offset_] *= val;
		}
	}

	return *this;
}

template <typename Scalar, int Rows, int Cols>
template <typename Scalar2>
CUDAH Matrix<Scalar, Rows, Cols>& Matrix<Scalar, Rows, Cols>::operator/=(Scalar2 val)
{
#pragma unroll
	for (int i = 0; i < Rows; i++) {
#pragma unroll
		for (int j = 0; j < Cols; j++) {
			buffer_[(i * Cols + j) * offset_] /= val;
		}
	}

	return *this;
}

template <typename Scalar, int Rows, int Cols>
CUDAH bool Matrix<Scalar, Rows, Cols>::transpose(Matrix<Scalar, Cols, Rows> &output) {
#pragma unroll
	for (int i = 0; i < Rows; i++) {
#pragma unroll
		for (int j = 0; j < Cols; j++) {
			output(j, i) = buffer_[(i * Cols + j) * offset_];
		}
	}

	return true;
}

//Only applicable for 3x3 matrix or below
template <typename Scalar, int Rows, int Cols>
CUDAH bool Matrix<Scalar, Rows, Cols>::inverse(Matrix<Scalar, Rows, Cols> &output) {
	return true;
}

//template <typename Scalar, int Rows, int Cols>
//CUDAH Scalar Matrix<Scalar, Rows, Cols>::dot(const Matrix<Scalar, Rows, Cols> &other)
//{
//	Scalar res = 0;
//
//#pragma unroll
//	for (int i = 0; i < Rows; i++) {
//#pragma unroll
//		for (int j = 0; j < Cols; j++) {
//			res += buffer_[(i * Rows + j) * offset_] * other(i, j);
//		}
//	}
//
//	return res;
//}

template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, Rows, 1> Matrix<Scalar, Rows, Cols>::col(int index) {
	return Matrix<Scalar, Rows, 1>(offset_ * Cols, buffer_ + index * offset_);
}

template <typename Scalar, int Rows, int Cols>
CUDAH Matrix<Scalar, 1, Cols> Matrix<Scalar, Rows, Cols>::row(int index) {
	return Matrix<Scalar, 1, Cols>(offset_, buffer_ + index * Cols * offset_);
}

template <typename Scalar, int Rows, int Cols>
template <int RSize>
CUDAH Matrix<Scalar, RSize, 1> Matrix<Scalar, Rows, Cols>::col(int row, int col)
{
	return Matrix<Scalar, RSize, 1>(offset_ * Cols, buffer_ + (row * Cols + col) * offset_);
}

template <typename Scalar, int Rows, int Cols>
template <int CSize>
CUDAH Matrix<Scalar, 1, CSize> Matrix<Scalar, Rows, Cols>::row(int row, int col)
{
	return Matrix<Scalar, 1, CSize>(offset_, buffer_ + (row * Cols + col) * offset_);
}

template <>
CUDAH bool Matrix<double, 1, 1>::inverse(Matrix<double, 1, 1> &output) {
	if (buffer_[0] != 0)
		output(0, 0) = 1 / buffer_[0];
	return true;
}

template <>
CUDAH bool Matrix<double, 2, 2>::inverse(Matrix<double, 2, 2> &output) {
	double det = at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);

	if (det != 0) {
		output(0, 0) = at(1, 1) / det;
		output(0, 1) = - at(0, 1) / det;

		output(1, 0) = - at(1, 0) / det;
		output(1, 1) = at(0, 0) / det;
	} else
		return false;

	return true;
}

template <>
CUDAH bool Matrix<double, 3, 3>::inverse(Matrix<double, 3, 3> &output) {
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

	return true;
}


template class Matrix<float, 3, 3>;
template class Matrix<double, 3, 3>;
template class Matrix<double, 6, 1>;
template class Matrix<double, 3, 6>;		// Point gradient class
template class Matrix<double, 18, 6>;		// Point hessian class
template class Matrix<double, 24, 1>;
template class Matrix<double, 45, 1>;

}

#endif
