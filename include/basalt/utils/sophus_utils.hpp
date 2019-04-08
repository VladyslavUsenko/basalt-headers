/**
BSD 3-Clause License

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <sophus/sim2.hpp>

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

namespace Eigen {

template <typename T>
using vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using map = std::map<K, V, std::less<K>,
                     Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using unordered_map =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       Eigen::aligned_allocator<std::pair<K const, V>>>;

}  // namespace Eigen

namespace Sophus {

template <typename Scalar>
inline static typename SE3<Scalar>::Tangent logd(const SE3<Scalar> &se3) {
  typename SE3<Scalar>::Tangent upsilon_omega;
  upsilon_omega.template tail<3>() = se3.so3().log();
  upsilon_omega.template head<3>() = se3.translation();

  return upsilon_omega;
}

template <typename Derived>
inline static SE3<typename Derived::Scalar> expd(
    const Eigen::MatrixBase<Derived> &upsilon_omega) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

  using Scalar = typename Derived::Scalar;

  return SE3<Scalar>(SO3<Scalar>::exp(upsilon_omega.template tail<3>()),
                     upsilon_omega.template head<3>());
}

// exp(phi+e) ~= exp(phi)*exp(J*e)
template <typename Derived1, typename Derived2>
void rightJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                      const Eigen::MatrixBase<Derived2> &J_const) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

  Scalar phi_norm2 = phi.squaredNorm();
  Scalar phi_norm = std::sqrt(phi_norm2);
  Scalar phi_norm3 = phi_norm2 * phi_norm;

  J.setIdentity();

  if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
    J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
  }
}

// log(exp(phi)exp(e)) ~= phi + J*e
template <typename Derived1, typename Derived2>
void rightJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                         const Eigen::MatrixBase<Derived2> &J_const) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

  Scalar phi_norm2 = phi.squaredNorm();
  Scalar phi_norm = std::sqrt(phi_norm2);

  J.setIdentity();

  if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J += phi_hat / 2;
    J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                         (2 * phi_norm * std::sin(phi_norm)));
  }
}

// exp(phi+e) ~= exp(J*e)*exp(phi)
template <typename Derived1, typename Derived2>
void leftJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                     const Eigen::MatrixBase<Derived2> &J_const) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

  Scalar phi_norm2 = phi.squaredNorm();
  Scalar phi_norm = std::sqrt(phi_norm2);
  Scalar phi_norm3 = phi_norm2 * phi_norm;

  J.setIdentity();

  if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
    J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
  }
}

// log(exp(e)*exp(phi)) ~= phi + J*e
template <typename Derived1, typename Derived2>
void leftJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                        const Eigen::MatrixBase<Derived2> &J_const) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

  Scalar phi_norm2 = phi.squaredNorm();
  Scalar phi_norm = std::sqrt(phi_norm2);

  J.setIdentity();

  if (Sophus::Constants<Scalar>::epsilon() < phi_norm) {
    Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
    Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

    J -= phi_hat / 2;
    J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                         (2 * phi_norm * std::sin(phi_norm)));
  }
}

// expd(phi+e) ~= expd(phi)*expd(J*e)
template <typename Derived1, typename Derived2>
void rightJacobianSE3Decoupled(const Eigen::MatrixBase<Derived1> &phi,
                               const Eigen::MatrixBase<Derived2> &J_const) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6);

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

  J.setZero();

  Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
  rightJacobianSO3(omega, J.template bottomRightCorner<3, 3>());
  J.template topLeftCorner<3, 3>() =
      Sophus::SO3<Scalar>::exp(omega).inverse().matrix();
}

// logd(expd(phi)expd(e)) ~= phi + J*e
template <typename Derived1, typename Derived2>
void rightJacobianInvSE3Decoupled(const Eigen::MatrixBase<Derived1> &phi,
                                  const Eigen::MatrixBase<Derived2> &J_const) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6);

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_const);

  J.setZero();

  Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
  rightJacobianInvSO3(omega, J.template bottomRightCorner<3, 3>());
  J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).matrix();
}

}  // namespace Sophus
