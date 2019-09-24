/* A lightweight class containing operations for 3x1 vectors
 * in the GPU kernel. These vectors are stored in
 * registers instead of arrays so performing operations
 * on them is faster than using normal MatrixDevice class.
 */
#ifndef VECTOR_REG_H_
#define VECTOR_REG_H_

#include <cuda.h>
#include "Matrix.h"
#include "MatrixDevice.h"
#include "MatrixDeviceList.h"

namespace gpu {
template <typename Scalar>
class VectorR {
public:
	CUDAH VectorR();

	// Copy constructors
	template <typename Scalar2>
	CUDAH VectorR(const VectorR<Scalar2> &other);

	template <typename Scalar2>
	CUDAH VectorR(const MatrixDevice<Scalar2, 3, 1> &other);

	template <typename Scalar2>
	CUDAH VectorR(const MatrixDevice<Scalar2, 1, 3> &other);

	// Move constructors
	template <typename Scalar2>
	CUDAH VectorR(VectorR<Scalar2> &&other);

	template <typename Scalar2>
	CUDAH VectorR(MatrixDevice<Scalar2, 3, 1> &&other);

	template <typename Scalar2>
	CUDAH VectorR(MatrixDevice<Scalar2, 1, 3> &&other);

	// Copy assignments
	template <typename Scalar2>
	CUDAH VectorR<Scalar> &operator=(const VectorR<Scalar2> &other);

	template <typename Scalar2>
	CUDAH VectorR<Scalar> &operator=(const MatrixDevice<Scalar2, 3, 1> &other);

	template <typename Scalar2>
	CUDAH VectorR<Scalar> &operator=(const MatrixDevice<Scalar2, 1, 3> &other);

	// Move assignments
	template <typename Scalar2>
	CUDAH VectorR<Scalar> &operator=(VectorR<Scalar2> &&other);

	template <typename Scalar2>
	CUDAH VectorR<Scalar> &operator=(MatrixDevice<Scalar2, 3, 1> &&other);

	template <typename Scalar2>
	CUDAH VectorR<Scalar> &operator=(MatrixDevice<Scalar2, 1, 3> &&other);

	CUDAH Scalar &operator()(int col);

	template <typename Scalar2>
	CUDAH Scalar dot(const VectorR<Scalar2> &other);

	template <typename Scalar2>
	CUDAH Scalar dot(const Matrix<Scalar2, 3, 1> &other);

	CUDAH VectorR<Scalar> &operator-=(const VectorR<Scalar> &other);
	CUDAH VectorR<Scalar> &operator+=(const VectorR<Scalar> &other);
private:
	Scalar x_, y_, z_;
};

template <typename Scalar>
CUDAH VectorR<Scalar>::VectorR()
{
	x_ = y_ = z_ = 0;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar>::VectorR(const VectorR<Scalar2> &other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar>::VectorR(const MatrixDevice<Scalar2, 3, 1> &other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar>::VectorR(const MatrixDevice<Scalar2, 1, 3> &other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar>::VectorR(VectorR<Scalar2> &&other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar>::VectorR(MatrixDevice<Scalar2, 3, 1> &&other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar>::VectorR(MatrixDevice<Scalar2, 1, 3> &&other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));
}


template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator=(const VectorR<Scalar2> &other)
{
	x_ = static_cast<Scalar>(other.x_);
	y_ = static_cast<Scalar>(other.y_);
	z_ = static_cast<Scalar>(other.z_);

	return *this;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator=(const MatrixDevice<Scalar2, 3, 1> &other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));

	return *this;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator=(const MatrixDevice<Scalar2, 1, 3> &other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));

	return *this;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator=(VectorR<Scalar2> &&other)
{
	x_ = static_cast<Scalar>(other.x_);
	y_ = static_cast<Scalar>(other.y_);
	z_ = static_cast<Scalar>(other.z_);

	return *this;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator=(MatrixDevice<Scalar2, 3, 1> &&other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));

	return *this;
}

template <typename Scalar>
template <typename Scalar2>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator=(MatrixDevice<Scalar2, 1, 3> &&other)
{
	x_ = static_cast<Scalar>(other(0));
	y_ = static_cast<Scalar>(other(1));
	z_ = static_cast<Scalar>(other(2));

	return *this;
}

template <typename Scalar>
CUDAH Scalar &VectorR<Scalar>::operator()(int col)
{
	switch (col) {
	case (0):
			return x_;
	case (1):
			return y_;
	default:
			return z_;
	}
}

template <typename Scalar>
template <typename Scalar2>
CUDAH Scalar VectorR<Scalar>::dot(const VectorR<Scalar2> &other)
{
	return x_ * static_cast<Scalar>(other.x_) + y_ * static_cast<Scalar>(other.y_) + z_ * static_cast<Scalar>(other.z_);
}

template <typename Scalar>
template <typename Scalar2>
CUDAH Scalar VectorR<Scalar>::dot(const Matrix<Scalar2, 3, 1> &other)
{
	return x_ * static_cast<Scalar>(other(0)) + y_ * static_cast<Scalar>(other(1)) + z_ * static_cast<Scalar>(other(2));
}

template <typename Scalar, typename Scalar2>
CUDAH Scalar dot(MatrixDevice<Scalar, 3, 1> &a, VectorR<Scalar2> &b)
{
	return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}

template <>
template <>
CUDAH double VectorR<double>::dot(const VectorR<double> &other)
{
	return x_ * other.x_ + y_ * other.y_ + z_ * other.z_;
}

template <>
template <>
CUDAH float VectorR<float>::dot(const VectorR<float> &other)
{
	return x_ * other.x_ + y_ * other.y_ + z_ * other.z_;
}

template <typename Scalar>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator-=(const VectorR<Scalar> &other)
{
	x_ -= other.x_;
	y_ -= other.y_;
	z_ -= other.z_;

	return *this;
}

template <typename Scalar>
CUDAH VectorR<Scalar> &VectorR<Scalar>::operator+=(const VectorR<Scalar> &other)
{
	x_ += other.x_;
	y_ += other.y_;
	z_ += other.z_;

	return *this;
}

template class VectorR<double>;
template class VectorR<float>;

}

#endif
