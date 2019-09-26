#ifndef MAScalarRIX_HOSScalar_H_
#define MAScalarRIX_HOSScalar_H_

#include "Matrix.h"
#include "MatrixDevice.h"

namespace gpu {
template <typename Scalar>
class MatrixHost : public Matrix<Scalar> {
public:
	MatrixHost();
	MatrixHost(int rows, int cols, int offset, Scalar *buffer);
	MatrixHost(const MatrixHost<Scalar> &other);
	MatrixHost(MatrixHost<Scalar> &&other);

	bool moveToGpu(MatrixDevice<Scalar> output);
	bool moveToHost(const MatrixDevice<Scalar> input);

	// Copy assignment
	MatrixHost<Scalar>& operator=(const MatrixHost<Scalar> &other);

	// Move assignment
	MatrixHost<Scalar>& operator=(MatrixHost<Scalar> &&other);

	void debug();

	template <typename Scalar2>
	friend std::ostream &operator<<(std::ostream &os, const MatrixHost<Scalar2> &value);

	~MatrixHost();

private:
	using Matrix<Scalar>::buffer_;
	using Matrix<Scalar>::offset_;
	using Matrix<Scalar>::is_copied_;
	using Matrix<Scalar>::rows_;
	using Matrix<Scalar>::cols_;
};



}

#endif
