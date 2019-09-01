/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

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

@file
@brief Uniform b-spline for euclidean vectors
*/

#pragma once

#include <basalt/spline/spline_common.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/sophus_utils.hpp>

#include <Eigen/Dense>

#include <array>

namespace basalt {

// Uniform b-spline for vectors with dimention _DIM of order _N
template <int _DIM, int _N, typename _Scalar = double>
class RdSpline {
 public:
  static constexpr int N = _N;
  static constexpr int DEG = _N - 1;

  static constexpr int DIM = _DIM;

  static constexpr _Scalar ns_to_s = 1e-9;
  static constexpr _Scalar s_to_ns = 1e9;

  using MatN = Eigen::Matrix<_Scalar, _N, _N>;
  using VecN = Eigen::Matrix<_Scalar, _N, 1>;

  using VecD = Eigen::Matrix<_Scalar, _DIM, 1>;
  using MatD = Eigen::Matrix<_Scalar, _DIM, _DIM>;

  struct JacobianStruct {
    size_t start_idx;
    std::array<_Scalar, N> d_val_d_knot;
  };

  RdSpline() : dt_ns(0), start_t_ns(0) {}

  RdSpline(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : dt_ns(time_interval_ns), start_t_ns(start_time_ns) {
    pow_inv_dt[0] = 1.0;
    pow_inv_dt[1] = s_to_ns / dt_ns;

    for (size_t i = 2; i < N; i++) {
      pow_inv_dt[i] = pow_inv_dt[i - 1] * pow_inv_dt[1];
    }
  }

  template <typename Scalar2>
  inline RdSpline<_DIM, _N, Scalar2> cast() const {
    RdSpline<_DIM, _N, Scalar2> res;

    res.dt_ns = dt_ns;
    res.start_t_ns = start_t_ns;

    for (int i = 0; i < _N; i++) res.pow_inv_dt[i] = pow_inv_dt[i];

    for (const auto k : knots)
      res.knots.emplace_back(k.template cast<Scalar2>());

    return res;
  }

  inline void setStartTimeNs(int64_t s) { start_t_ns = s; }

  int64_t maxTimeNs() const {
    return start_t_ns + (knots.size() - N + 1) * dt_ns - 1;
  }

  int64_t minTimeNs() const { return start_t_ns; }

  void genRandomTrajectory(int n, bool static_init = false) {
    if (static_init) {
      VecD rnd = VecD::Random() * 5;

      for (int i = 0; i < N; i++) knots.push_back(rnd);
      for (int i = 0; i < n - N; i++) knots.push_back(VecD::Random() * 5);
    } else {
      for (int i = 0; i < n; i++) knots.push_back(VecD::Random() * 5);
    }
  }

  inline void knots_push_back(const VecD& knot) { knots.push_back(knot); }
  inline void knots_pop_back() { knots.pop_back(); }
  inline const VecD& knots_front() const { return knots.front(); }
  inline void knots_pop_front() {
    start_t_ns += dt_ns;
    knots.pop_front();
  }

  inline void resize(size_t n) { knots.resize(n); }

  inline VecD& getKnot(int i) { return knots[i]; }
  inline const VecD& getKnot(int i) const { return knots[i]; }

  const Eigen::deque<VecD>& getKnots() const { return knots; }

  int64_t getTimeIntervalNs() const { return dt_ns; }
  int64_t getStartTimeNs() const { return start_t_ns; }

  template <int Derivative = 0>
  VecD evaluate(int64_t time_ns, JacobianStruct* J = nullptr) const {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    VecN p;
    baseCoeffsWithTime<Derivative>(p, u);

    VecN coeff = pow_inv_dt[Derivative] * (blending_matrix_ * p);

    // std::cerr << "p " << p.transpose() << std::endl;
    // std::cerr << "coeff " << coeff.transpose() << std::endl;

    VecD res;
    res.setZero();

    for (int i = 0; i < N; i++) {
      res += coeff[i] * knots[s + i];

      if (J) J->d_val_d_knot[i] = coeff[i];
    }

    if (J) J->start_idx = s;

    return res;
  }

  inline VecD velocity(int64_t time_ns, JacobianStruct* J = nullptr) const {
    return evaluate<1>(time_ns, J);
  }

  inline VecD acceleration(int64_t time_ns, JacobianStruct* J = nullptr) const {
    return evaluate<2>(time_ns, J);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  template <int Derivative, class Derived>
  static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const,
                                 _Scalar t) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
    Eigen::MatrixBase<Derived>& res =
        const_cast<Eigen::MatrixBase<Derived>&>(res_const);

    res.setZero();

    if (Derivative < N) {
      res[Derivative] = base_coefficients_(Derivative, Derivative);

      _Scalar _t = t;
      for (int j = Derivative + 1; j < N; j++) {
        res[j] = base_coefficients_(Derivative, j) * _t;
        _t = _t * t;
      }
    }
  }

  template <int, int, typename>
  friend class RdSpline;

  static const MatN blending_matrix_;
  static const MatN base_coefficients_;

  int64_t dt_ns;

  Eigen::deque<VecD> knots;
  int64_t start_t_ns;
  std::array<_Scalar, _N> pow_inv_dt;
};

template <int _DIM, int _N, typename _Scalar>
const typename RdSpline<_DIM, _N, _Scalar>::MatN
    RdSpline<_DIM, _N, _Scalar>::base_coefficients_ =
        computeBaseCoefficients<_N, _Scalar>();

template <int _DIM, int _N, typename _Scalar>
const typename RdSpline<_DIM, _N, _Scalar>::MatN
    RdSpline<_DIM, _N, _Scalar>::blending_matrix_ =
        computeBlendingMatrix<_N, _Scalar, false>();

}  // namespace basalt
