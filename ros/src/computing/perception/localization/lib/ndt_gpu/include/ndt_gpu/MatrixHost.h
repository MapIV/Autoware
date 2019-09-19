#ifndef MATRIX_HOST_H_
#define MATRIX_HOST_H_

#include "Matrix.h"
#include "MatrixDevice.h"

namespace gpu {
template <typename T>
class MatrixHost : public Matrix<T> {
public:
	MatrixHost();
	MatrixHost(int rows, int cols);
	MatrixHost(int rows, int cols, int offset, const T *buffer);
	MatrixHost(const MatrixHost<T> &other);
	MatrixHost(MatrixHost<T> &&other);

	bool moveToGpu(MatrixDevice<T> output);
	bool moveToHost(const MatrixDevice<T> input);

	// Copy assignment
	MatrixHost<T>& operator=(const MatrixHost<T> &other);

	// Move assignment
	MatrixHost<T>& operator=(MatrixHost<T> &&other);

	void debug();

	friend std::ostream &operator<<(std::ostream &os, const MatrixHost<T> &value);

	~MatrixHost();
private:
	bool fr_;
};

}

#endif
