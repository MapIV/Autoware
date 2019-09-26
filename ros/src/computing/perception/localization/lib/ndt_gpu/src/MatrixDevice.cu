#include "ndt_gpu/MatrixDevice.h"
#include "ndt_gpu/debug.h"

namespace gpu {
template <typename Scalar>
MatrixDevice<Scalar>::MatrixDevice()
{
	offset_ = 0;
	rows_ = cols_ = 0;
	buffer_ = NULL;
	is_copied_ = false;
}

template <typename Scalar>
MatrixDevice<Scalar>::MatrixDevice(int rows, int cols)
{
	offset_ = 1;
	rows_ = rows;
	cols_ = cols;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(Scalar) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(Scalar) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaDeviceSynchronize());

	is_copied_ = false;
}

template <typename Scalar>
void MatrixDevice<Scalar>::free()
{
	if (!is_copied_ && buffer_ != NULL) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = NULL;
	}
}

template <typename Scalar>
MatrixDevice<Scalar>& MatrixDevice<Scalar>::operator=(const MatrixDevice<Scalar> &other)
{
	if (!is_copied_ && buffer_ != NULL) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = NULL;
	}

	buffer_ = other.buffer_;
	offset_ = other.offset_;
	rows_ = other.rows_;
	cols_ = other.cols_;
	is_copied_ = true;

	return *this;
}

template <typename Scalar>
MatrixDevice<Scalar>& MatrixDevice<Scalar>::operator=(MatrixDevice<Scalar> &&other)
{
	if (!is_copied_ && buffer_ != NULL) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = NULL;
	}

	buffer_ = other.buffer_;
	is_copied_ = false;
	offset_ = other.offset_;
	rows_ = other.rows_;
	cols_ = other.cols_;

	other.buffer_ = NULL;
	other.is_copied_ = true;
	other.offset_ = 0;

	return *this;
}

template class MatrixDevice<float>;
template class MatrixDevice<double>;

}
