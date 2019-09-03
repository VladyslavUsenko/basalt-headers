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

#include <basalt/spline/rd_spline.h>
#include <basalt/spline/so3_spline.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

template <int DIM, int N, int DERIV>
void test_evaluate(const basalt::RdSpline<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdSpline<DIM, N>::VecD;
  using MatrixD = typename basalt::RdSpline<DIM, N>::MatD;

  typename basalt::RdSpline<DIM, N>::JacobianStruct J;

  spline.template evaluate<DERIV>(t_ns, &J);

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_val_d_knot" << i << " time " << t_ns;

    MatrixD Ja;
    Ja.setZero();

    if (i >= J.start_idx && i < J.start_idx + N) {
      Ja.diagonal().setConstant(J.d_val_d_knot[i - J.start_idx]);
    }

    test_jacobian(
        ss.str(), Ja,
        [&](const VectorD &x) {
          basalt::RdSpline<DIM, N> spline1 = spline;
          spline1.getKnot(i) += x;

          return spline1.template evaluate<DERIV>(t_ns);
        },
        x0);
  }
}

template <int DIM, int N, int DERIV>
void test_time_deriv(const basalt::RdSpline<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdSpline<DIM, N>::VecD;

  VectorD d_val_d_t = spline.template evaluate<DERIV + 1>(t_ns);

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_val_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return spline.template evaluate<DERIV>(t_ns + inc);
      },
      x0);
}

template <int N>
void test_evaluate_so3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::VecD;
  using MatrixD = typename basalt::So3Spline<5>::MatD;
  using SO3 = typename basalt::So3Spline<5>::SO3;

  typename basalt::So3Spline<N>::JacobianStruct J;

  SO3 res = spline.evaluate(t_ns, &J);

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_val_d_knot" << i << " time " << t_ns;

    MatrixD Ja;
    Ja.setZero();

    if (i >= J.start_idx && i < J.start_idx + N) {
      Ja = J.d_val_d_knot[i - J.start_idx];
    }

    test_jacobian(
        ss.str(), Ja,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);

          SO3 res1 = spline1.evaluate(t_ns);

          return (res1 * res.inverse()).log();
        },
        x0);
  }
}

template <int N>
void test_vel_so3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::VecD;
  using SO3 = typename basalt::So3Spline<5>::SO3;

  SO3 res = spline.evaluate(t_ns);

  VectorD d_res_d_t = spline.velocityBody(t_ns);

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return (res.inverse() * spline.evaluate(t_ns + inc)).log();
      },
      x0);
}

template <int N>
void test_evaluate_so3_vel(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::VecD;
  using MatrixD = typename basalt::So3Spline<5>::MatD;
  using SO3 = typename basalt::So3Spline<5>::SO3;

  typename basalt::So3Spline<N>::JacobianStruct J;

  VectorD res = spline.velocityBody(t_ns, &J);
  VectorD res_ref = spline.velocityBody(t_ns);

  ASSERT_TRUE(res_ref.isApprox(res)) << "res and res_ref are not the same";

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;

    ss << "d_vel_d_knot" << i << " time " << t_ns;

    MatrixD Ja;
    Ja.setZero();

    if (i >= J.start_idx && i < J.start_idx + N) {
      Ja = J.d_val_d_knot[i - J.start_idx];
    }

    test_jacobian(
        ss.str(), Ja,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);

          return spline1.velocityBody(t_ns);
        },
        x0);
  }
}

TEST(SplineTest, UBSplineEvaluateKnots) {
  static const int DIM = 3;
  static const int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8)
    test_evaluate<DIM, N, 0>(spline, t_ns);
}

TEST(SplineTest, UBSplineVelocityKnots) {
  static const int DIM = 3;
  static const int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8)
    test_evaluate<DIM, N, 1>(spline, t_ns);
}

TEST(SplineTest, UBSplineAccelKnots) {
  static const int DIM = 3;
  static const int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8)
    test_evaluate<DIM, N, 2>(spline, t_ns);
}

TEST(SplineTest, UBSplineEvaluateTimeDeriv) {
  static const int DIM = 3;
  static const int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8)
    test_time_deriv<DIM, N, 0>(spline, t_ns);
}

TEST(SplineTest, UBSplineVelocityTimeDeriv) {
  static const int DIM = 3;
  static const int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8)
    test_time_deriv<DIM, N, 1>(spline, t_ns);
}

TEST(SplineTest, SO3CUBSplineEvaluateKnots) {
  static const int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8)
    test_evaluate_so3<N>(spline, t_ns);
}

TEST(SplineTest, SO3CUBSplineVelocity) {
  static const int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8)
    test_vel_so3<5>(spline, t_ns);
}

TEST(SplineTest, SO3CUBSplineVelocityKnots) {
  static const int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8)
    test_evaluate_so3_vel<5>(spline, t_ns);
}

TEST(SplineTest, SO3CUBSplineBounds) {
  static const int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // std::cerr << "spline.maxTimeNs() " << spline.maxTimeNs() << std::endl;

  spline.evaluate(spline.maxTimeNs());
  // std::cerr << "res1\n" << res1.matrix() << std::endl;
  spline.evaluate(spline.minTimeNs());
  // std::cerr << "res2\n" << res2.matrix() << std::endl;

  // Sophus::SO3d res3 = spline.evaluate(spline.maxTimeNs() + 1);
  // std::cerr << "res3\n" << res1.matrix() << std::endl;
  // Sophus::SO3d res4 = spline.evaluate(spline.minTimeNs() - 1);
  // std::cerr << "res4\n" << res2.matrix() << std::endl;
}

TEST(SplineTest, UBSplineBounds) {
  static const int N = 5;
  static const int DIM = 3;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // std::cerr << "spline.maxTimeNs() " << spline.maxTimeNs() << std::endl;

  spline.evaluate(spline.maxTimeNs());
  // std::cerr << "res1\n" << res1.matrix() << std::endl;
  spline.evaluate(spline.minTimeNs());
  // std::cerr << "res2\n" << res2.matrix() << std::endl;

  // Eigen::Vector3d res3 = spline.evaluate(spline.maxTimeNs() + 1);
  // std::cerr << "res3\n" << res1.matrix() << std::endl;
  // Eigen::Vector3d res4 = spline.evaluate(spline.minTimeNs() - 1);
  // std::cerr << "res4\n" << res2.matrix() << std::endl;
}
