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
@brief Uniform cumulative b-spline for SO(3)
*/

#pragma once

#include <basalt/spline/spline_common.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/sophus_utils.hpp>

#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include <array>

namespace basalt {

// SO(3) cumulative uniform b-spline of order _N
template <int _N, typename _Scalar = double>
class So3Spline {
 public:
  static constexpr int N = _N;
  static constexpr int DEG = _N - 1;

  static constexpr _Scalar ns_to_s = 1e-9;
  static constexpr _Scalar s_to_ns = 1e9;

  using MatN = Eigen::Matrix<_Scalar, _N, _N>;
  using VecN = Eigen::Matrix<_Scalar, _N, 1>;

  using VecD = Eigen::Matrix<_Scalar, 3, 1>;
  using MatD = Eigen::Matrix<_Scalar, 3, 3>;

  using SO3 = Sophus::SO3<_Scalar>;

  struct JacobianStruct {
    size_t start_idx;
    std::array<MatD, _N> d_val_d_knot;
  };

  So3Spline(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : dt_ns(time_interval_ns), start_t_ns(start_time_ns) {
    pow_inv_dt[0] = 1.0;
    pow_inv_dt[1] = s_to_ns / dt_ns;
  }

  int64_t maxTimeNs() const {
    return start_t_ns + (knots.size() - N + 1) * dt_ns - 1;
  }

  int64_t minTimeNs() const { return start_t_ns; }

  void genRandomTrajectory(int n, bool static_init = false) {
    if (static_init) {
      VecD rnd = VecD::Random() * M_PI;

      for (int i = 0; i < N; i++) knots.push_back(SO3::exp(rnd));

      for (int i = 0; i < n - N; i++)
        knots.push_back(SO3::exp(VecD::Random() * M_PI));

    } else {
      for (int i = 0; i < n; i++)
        knots.push_back(SO3::exp(VecD::Random() * M_PI));
    }
  }

  inline void setStartTimeNs(int64_t s) { start_t_ns = s; }

  inline void knots_push_back(const SO3& knot) { knots.push_back(knot); }
  inline void knots_pop_back() { knots.pop_back(); }
  inline const SO3& knots_front() const { return knots.front(); }
  inline void knots_pop_front() {
    start_t_ns += dt_ns;
    knots.pop_front();
  }

  inline void resize(size_t n) { knots.resize(n); }

  inline SO3& getKnot(int i) { return knots[i]; }

  inline const SO3& getKnot(int i) const { return knots[i]; }

  const Eigen::deque<SO3>& getKnots() const { return knots; }

  int64_t getTimeIntervalNs() const { return dt_ns; }

  SO3 evaluate(int64_t time_ns, JacobianStruct* J = nullptr) const {
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
    baseCoeffsWithTime<0>(p, u);

    VecN coeff = blending_matrix_ * p;

    SO3 res = knots[s];

    MatD J_helper;

    if (J) {
      J->start_idx = s;
      J_helper.setIdentity();
    }

    for (int i = 0; i < DEG; i++) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      Sophus::SO3d r01 = p0.inverse() * p1;
      Eigen::Vector3d delta = r01.log();
      Eigen::Vector3d kdelta = delta * coeff[i + 1];

      if (J) {
        Eigen::Matrix3d Jl_inv_delta, Jl_k_delta;

        Sophus::leftJacobianInvSO3(delta, Jl_inv_delta);
        Sophus::leftJacobianSO3(kdelta, Jl_k_delta);

        J->d_val_d_knot[i] = J_helper;
        J_helper = coeff[i + 1] * res.matrix() * Jl_k_delta * Jl_inv_delta *
                   p0.inverse().matrix();
        J->d_val_d_knot[i] -= J_helper;
      }
      res *= Sophus::SO3d::exp(kdelta);
    }

    if (J) J->d_val_d_knot[DEG] = J_helper;

    return res;
  }

  VecD velocityBody(int64_t time_ns) const {
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
    baseCoeffsWithTime<0>(p, u);
    VecN coeff = blending_matrix_ * p;

    baseCoeffsWithTime<1>(p, u);
    VecN dcoeff = pow_inv_dt[1] * blending_matrix_ * p;

    SO3 r_accum;

    VecD res;
    res.setZero();

    for (int i = DEG - 1; i >= 0; i--) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      VecD delta = r01.log();

      res += r_accum.inverse().matrix() * delta * dcoeff[i + 1];
      r_accum = SO3::exp(delta * coeff[i + 1]) * r_accum;
    }

    return res;
  }

  VecD velocityBody(int64_t time_ns, JacobianStruct* J) const {
    BASALT_ASSERT(J);

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
    baseCoeffsWithTime<0>(p, u);
    VecN coeff = blending_matrix_ * p;

    baseCoeffsWithTime<1>(p, u);
    VecN dcoeff = pow_inv_dt[1] * blending_matrix_ * p;

    VecD res;
    res.setZero();

    VecD delta_vec[DEG];
    VecD k_delta_vec[DEG];

    MatD R_tmp[DEG];
    SO3 accum;
    SO3 exp_k_delta[DEG];

    MatD Jr_delta_inv[DEG], JrkJri[DEG];

    for (int i = DEG - 1; i >= 0; i--) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      delta_vec[i] = r01.log();

      Sophus::rightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
      Jr_delta_inv[i] *= p1.inverse().matrix();

      k_delta_vec[i] = coeff[i + 1] * delta_vec[i];
      Sophus::rightJacobianSO3(-k_delta_vec[i], JrkJri[i]);
      JrkJri[i] *= Jr_delta_inv[i];

      res += accum * delta_vec[i] * dcoeff[i + 1];
      R_tmp[i] = accum.matrix();
      exp_k_delta[i] = Sophus::SO3d::exp(-k_delta_vec[i]);
      accum *= exp_k_delta[i];
    }

    MatD d_res_d_delta[DEG];

    d_res_d_delta[0] = R_tmp[0] * Jr_delta_inv[0] * dcoeff[1];
    Eigen::Vector3d v = delta_vec[0] * dcoeff[1];
    for (int i = 1; i < DEG; i++) {
      Eigen::Matrix3d tmp = Sophus::SO3d::hat(v) * JrkJri[i];

      d_res_d_delta[i] = R_tmp[i - 1] * tmp * coeff[i + 1] +
                         R_tmp[i] * Jr_delta_inv[i] * dcoeff[i + 1];

      v = exp_k_delta[i] * v + delta_vec[i] * dcoeff[i + 1];
    }

    J->start_idx = s;

    for (int i = 0; i < N; i++) J->d_val_d_knot[i].setZero();

    for (int i = 0; i < DEG; i++) {
      J->d_val_d_knot[i] -= d_res_d_delta[i];
      J->d_val_d_knot[i + 1] += d_res_d_delta[i];
    }

    return res;
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

  static const MatN blending_matrix_;
  static const MatN base_coefficients_;

  int64_t dt_ns;

  Eigen::deque<SO3> knots;
  int64_t start_t_ns;
  std::array<_Scalar, 2> pow_inv_dt;
};

template <int _N, typename _Scalar>
const typename So3Spline<_N, _Scalar>::MatN
    So3Spline<_N, _Scalar>::base_coefficients_ =
        computeBaseCoefficients<_N, _Scalar>();

template <int _N, typename _Scalar>
const typename So3Spline<_N, _Scalar>::MatN
    So3Spline<_N, _Scalar>::blending_matrix_ =
        computeBlendingMatrix<_N, _Scalar, true>();

}  // namespace basalt
