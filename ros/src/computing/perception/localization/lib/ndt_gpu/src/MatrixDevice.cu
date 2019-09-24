#include "ndt_gpu/MatrixDevice.h"
#include "ndt_gpu/debug.h"

namespace gpu {
template <typename Scalar, int Rows, int Cols>
MatrixDevice<Scalar, Rows, Cols>::MatrixDevice()
{
	if (buffer_ != NULL && fr_) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = NULL;
	}

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(Scalar) * Rows * Cols * offset_));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(Scalar) * Rows * Cols * offset_));
	checkCudaErrors(cudaDeviceSynchronize());
	fr_ = true;
}

template <typename Scalar, int Rows, int Cols>
void MatrixDevice<Scalar, Rows, Cols>::memFree()
{
	if (fr_ && buffer_ != NULL) {
			checkCudaErrors(cudaFree(buffer_));
			buffer_ = NULL;
	}
}

template <typename Scalar, int Rows, int Cols>
MatrixDevice<Scalar, Rows, Cols>& MatrixDevice<Scalar, Rows, Cols>::operator=(MatrixDevice<Scalar, Rows, Cols> &&other)
{
	if (fr_ && buffer_ != NULL) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = NULL;
	}

	buffer_ = other.buffer_;
	fr_ = other.fr_;
	offset_ = other.offset_;

	other.buffer_ = NULL;
	other.fr_ = false;
	other.offset_ = 0;

	return *this;
}

template class MatrixDevice<float, 3, 3>;
template class MatrixDevice<double, 3, 3>;
template class MatrixDevice<double, 3, 1>;
template class MatrixDevice<double, 6, 1>;
template class MatrixDevice<double, 3, 6>;		// Point gradient class
template class MatrixDevice<double, 18, 6>;		// Point hessian class
template class MatrixDevice<double, 24, 1>;
template class MatrixDevice<double, 45, 1>;


}
