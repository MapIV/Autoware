#include "ndt_gpu/MatrixDevice.h"
#include "ndt_gpu/debug.h"

namespace gpu {
template <typename T>
void MatrixDevice<T>::memAlloc()
{
	if (buffer_ != NULL && fr_) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = NULL;
	}

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(double) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(double) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaDeviceSynchronize());
	fr_ = true;
}

template <typename T>
void MatrixDevice<T>::memFree()
{
	if (fr_) {
		if (buffer_ != NULL) {
			checkCudaErrors(cudaFree(buffer_));
			buffer_ = NULL;
		}
	}
}

template <typename T>
SquareMatrixDevice<T>::SquareMatrixDevice(int size) :
	MatrixDevice(size, size)
{}

}
