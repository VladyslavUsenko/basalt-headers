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
*/

#pragma once

#include <basalt/spline/rd_spline.h>
#include <basalt/spline/so3_spline.h>
#include <basalt/utils/assert.h>

#include <basalt/calibration/calib_bias.hpp>

#include <array>

namespace basalt {

// SE(3) uniform b-spline of order _N
template <int _N, typename _Scalar = double>
class Se3Spline {
 public:
  static constexpr int N = _N;
  static constexpr int DEG = _N - 1;

  using MatN = Eigen::Matrix<_Scalar, _N, _N>;
  using VecN = Eigen::Matrix<_Scalar, _N, 1>;
  using VecNp1 = Eigen::Matrix<_Scalar, _N + 1, 1>;

  using Vec3 = Eigen::Matrix<_Scalar, 3, 1>;
  using Vec6 = Eigen::Matrix<_Scalar, 6, 1>;
  using Vec9 = Eigen::Matrix<_Scalar, 9, 1>;
  using Vec12 = Eigen::Matrix<_Scalar, 12, 1>;

  using Mat3 = Eigen::Matrix<_Scalar, 3, 3>;
  using Mat6 = Eigen::Matrix<_Scalar, 6, 6>;

  using Mat36 = Eigen::Matrix<_Scalar, 3, 6>;
  using Mat39 = Eigen::Matrix<_Scalar, 3, 9>;
  using Mat312 = Eigen::Matrix<_Scalar, 3, 12>;

  using Matrix3Array = std::array<Mat3, N>;
  using Matrix36Array = std::array<Mat36, N>;
  using Matrix6Array = std::array<Mat6, N>;

  using SO3 = Sophus::SO3<_Scalar>;
  using SE3 = Sophus::SE3<_Scalar>;

  using PosJacobianStruct = typename RdSpline<3, N, _Scalar>::JacobianStruct;
  using SO3JacobianStruct = typename So3Spline<N, _Scalar>::JacobianStruct;

  struct AccelPosSO3JacobianStruct {
    size_t start_idx;
    std::array<Mat36, N> d_val_d_knot;
  };

  struct PosePosSO3JacobianStruct {
    size_t start_idx;
    std::array<Mat6, N> d_val_d_knot;
  };

  Se3Spline(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : pos_spline(time_interval_ns, start_time_ns),
        so3_spline(time_interval_ns, start_time_ns),
        dt_ns(time_interval_ns) {}

  Se3Spline(_Scalar time_interval, _Scalar start_time = 0) = delete;

  void genRandomTrajectory(int n, bool static_init = false) {
    so3_spline.genRandomTrajectory(n, static_init);
    pos_spline.genRandomTrajectory(n, static_init);
  }

  void setKnot(const Sophus::SE3d &pose, int i) {
    so3_spline.getKnot(i) = pose.so3();
    pos_spline.getKnot(i) = pose.translation();
  }

  void setKnots(const Sophus::SE3d &pose, int num_knots) {
    so3_spline.resize(num_knots);
    pos_spline.resize(num_knots);

    for (int i = 0; i < num_knots; i++) {
      so3_spline.getKnot(i) = pose.so3();
      pos_spline.getKnot(i) = pose.translation();
    }
  }

  void setKnots(const Se3Spline<N, _Scalar> &other) {
    BASALT_ASSERT(other.dt_ns == dt_ns);
    BASALT_ASSERT(other.pos_spline.getKnots().size() ==
                  other.pos_spline.getKnots().size());

    size_t num_knots = other.pos_spline.getKnots().size();

    so3_spline.resize(num_knots);
    pos_spline.resize(num_knots);

    for (size_t i = 0; i < num_knots; i++) {
      so3_spline.getKnot(i) = other.so3_spline.getKnot(i);
      pos_spline.getKnot(i) = other.pos_spline.getKnot(i);
    }
  }

  inline void knots_push_back(const SE3 &knot) {
    so3_spline.knots_push_back(knot.so3());
    pos_spline.knots_push_back(knot.translation());
  }

  inline void knots_pop_back() {
    so3_spline.knots_pop_back();
    pos_spline.knots_pop_back();
  }

  inline SE3 knots_front() const {
    SE3 res(so3_spline.knots_front(), pos_spline.knots_front());

    return res;
  }
  inline void knots_pop_front() {
    so3_spline.knots_pop_front();
    pos_spline.knots_pop_front();

    BASALT_ASSERT(so3_spline.minTimeNs() == pos_spline.minTimeNs());
    BASALT_ASSERT(so3_spline.getKnots().size() == pos_spline.getKnots().size());
  }

  SE3 getLastKnot() {
    BASALT_ASSERT(so3_spline.getKnots().size() == pos_spline.getKnots().size());

    SE3 res(so3_spline.getKnots().back(), pos_spline.getKnots().back());

    return res;
  }

  SE3 getKnot(size_t i) const {
    SE3 res(getKnotSO3(i), getKnotPos(i));
    return res;
  }

  inline SO3 &getKnotSO3(size_t i) { return so3_spline.getKnot(i); }

  inline const SO3 &getKnotSO3(size_t i) const { return so3_spline.getKnot(i); }

  inline Vec3 &getKnotPos(size_t i) { return pos_spline.getKnot(i); }

  inline const Vec3 &getKnotPos(size_t i) const {
    return pos_spline.getKnot(i);
  }

  inline void setStartTimeNs(int64_t s) {
    so3_spline.setStartTimeNs(s);
    pos_spline.setStartTimeNs(s);
  }

  template <typename Derived>
  void applyInc(int i, const Eigen::MatrixBase<Derived> &inc) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

    pos_spline.getKnot(i) += inc.template head<3>();
    so3_spline.getKnot(i) =
        SO3::exp(inc.template tail<3>()) * so3_spline.getKnot(i);
  }

  int64_t maxTimeNs() const {
    BASALT_ASSERT_STREAM(so3_spline.maxTimeNs() == pos_spline.maxTimeNs(),
                         "so3_spline.maxTimeNs() " << so3_spline.maxTimeNs()
                                                   << " pos_spline.maxTimeNs() "
                                                   << pos_spline.maxTimeNs());
    return pos_spline.maxTimeNs();
  }

  int64_t minTimeNs() const {
    BASALT_ASSERT_STREAM(so3_spline.minTimeNs() == pos_spline.minTimeNs(),
                         "so3_spline.minTimeNs() " << so3_spline.minTimeNs()
                                                   << " pos_spline.minTimeNs() "
                                                   << pos_spline.minTimeNs());
    return pos_spline.minTimeNs();
  }

  size_t numKnots() const { return pos_spline.getKnots().size(); }

  inline Vec3 transAccelWorld(int64_t time_ns) const {
    return pos_spline.acceleration(time_ns);
  }

  inline Vec3 transVelWorld(int64_t time_ns) const {
    return pos_spline.velocity(time_ns);
  }

  inline Vec3 rotVelBody(int64_t time_ns) const {
    return so3_spline.velocityBody(time_ns);
  }

  SE3 pose(int64_t time_ns) const {
    SE3 res;

    res.so3() = so3_spline.evaluate(time_ns);
    res.translation() = pos_spline.evaluate(time_ns);

    return res;
  }

  Sophus::SE3d pose(int64_t time_ns, PosePosSO3JacobianStruct *J) const {
    Sophus::SE3d res;

    typename So3Spline<_N, _Scalar>::JacobianStruct Jr;
    typename RdSpline<3, N, _Scalar>::JacobianStruct Jp;

    res.so3() = so3_spline.evaluate(time_ns, &Jr);
    res.translation() = pos_spline.evaluate(time_ns, &Jp);

    if (J) {
      Eigen::Matrix3d RT = res.so3().inverse().matrix();

      J->start_idx = Jr.start_idx;
      for (int i = 0; i < N; i++) {
        J->d_val_d_knot[i].setZero();
        J->d_val_d_knot[i].template topLeftCorner<3, 3>() =
            RT * Jp.d_val_d_knot[i];
        J->d_val_d_knot[i].template bottomRightCorner<3, 3>() =
            RT * Jr.d_val_d_knot[i];
      }
    }

    return res;
  }

  void d_pose_d_t(int64_t time_ns, Vec6 &J) const {
    J.template head<3>() =
        so3_spline.evaluate(time_ns).inverse() * transVelWorld(time_ns);
    J.template tail<3>() = rotVelBody(time_ns);
  }

  Vec3 gyroResidual(int64_t time_ns, const Vec3 &measurement,
                    const CalibGyroBias<_Scalar> &gyro_bias_full) const {
    return so3_spline.velocityBody(time_ns) -
           gyro_bias_full.getCalibrated(measurement);
  }

  Vec3 gyroResidual(int64_t time_ns, const Vec3 &measurement,
                    const CalibGyroBias<_Scalar> &gyro_bias_full,
                    SO3JacobianStruct *J_knots,
                    Mat312 *J_bias = nullptr) const {
    if (J_bias) {
      J_bias->setZero();
      J_bias->template block<3, 3>(0, 0).diagonal().array() = 1.0;
      J_bias->template block<3, 3>(0, 3).diagonal().array() = -measurement[0];
      J_bias->template block<3, 3>(0, 6).diagonal().array() = -measurement[1];
      J_bias->template block<3, 3>(0, 9).diagonal().array() = -measurement[2];
    }

    return so3_spline.velocityBody(time_ns, J_knots) -
           gyro_bias_full.getCalibrated(measurement);
  }

  Vec3 accelResidual(int64_t time_ns, const Eigen::Vector3d &measurement,
                     const CalibAccelBias<_Scalar> &accel_bias_full,
                     const Eigen::Vector3d &g) const {
    Sophus::SO3d R = so3_spline.evaluate(time_ns);
    Eigen::Vector3d accel_world = pos_spline.acceleration(time_ns);

    return R.inverse() * (accel_world + g) -
           accel_bias_full.getCalibrated(measurement);
  }

  Vec3 accelResidual(int64_t time_ns, const Vec3 &measurement,
                     const CalibAccelBias<_Scalar> &accel_bias_full,
                     const Vec3 &g, AccelPosSO3JacobianStruct *J_knots,
                     Mat39 *J_bias = nullptr, Mat3 *J_g = nullptr) const {
    typename So3Spline<_N, _Scalar>::JacobianStruct Jr;
    typename RdSpline<3, N, _Scalar>::JacobianStruct Jp;

    Sophus::SO3d R = so3_spline.evaluate(time_ns, &Jr);
    Eigen::Vector3d accel_world = pos_spline.acceleration(time_ns, &Jp);

    Eigen::Matrix3d RT = R.inverse().matrix();
    Eigen::Matrix3d tmp = RT * Sophus::SO3d::hat(accel_world + g);

    BASALT_ASSERT_STREAM(
        Jr.start_idx == Jp.start_idx,
        "Jr.start_idx " << Jr.start_idx << " Jp.start_idx " << Jp.start_idx);

    BASALT_ASSERT_STREAM(
        so3_spline.getKnots().size() == pos_spline.getKnots().size(),
        "so3_spline.getKnots().size() " << so3_spline.getKnots().size()
                                        << " pos_spline.getKnots().size() "
                                        << pos_spline.getKnots().size());

    J_knots->start_idx = Jp.start_idx;
    for (int i = 0; i < N; i++) {
      J_knots->d_val_d_knot[i].template topLeftCorner<3, 3>() =
          RT * Jp.d_val_d_knot[i];
      J_knots->d_val_d_knot[i].template bottomRightCorner<3, 3>() =
          tmp * Jr.d_val_d_knot[i];
    }

    if (J_bias) {
      J_bias->setZero();
      J_bias->template block<3, 3>(0, 0).diagonal().array() = 1.0;
      J_bias->template block<3, 3>(0, 3).diagonal().array() = -measurement[0];
      (*J_bias)(1, 6) = -measurement[1];
      (*J_bias)(2, 7) = -measurement[1];
      (*J_bias)(2, 8) = -measurement[2];
    }
    if (J_g) (*J_g) = RT;

    Vec3 res =
        RT * (accel_world + g) - accel_bias_full.getCalibrated(measurement);

    return res;
  }

  Sophus::Vector3d positionResidual(int64_t time_ns,
                                    const Vec3 &measured_position,
                                    PosJacobianStruct *Jp = nullptr) const {
    return pos_spline.evaluate(time_ns, Jp) - measured_position;
  }

  Sophus::Vector3d orientationResidual(int64_t time_ns,
                                       const SO3 &measured_orientation,
                                       SO3JacobianStruct *Jr = nullptr) const {
    Sophus::Vector3d res =
        (so3_spline.evaluate(time_ns, Jr) * measured_orientation.inverse())
            .log();

    if (Jr) {
      Eigen::Matrix3d Jrot;
      Sophus::leftJacobianSO3(res, Jrot);

      for (int i = 0; i < N; i++) {
        Jr->d_val_d_knot[i] = Jrot * Jr->d_val_d_knot[i];
      }
    }

    return res;
  }

  inline void print_knots() const {
    for (size_t i = 0; i < pos_spline.getKnots().size(); i++) {
      std::cerr << i << ": p:" << pos_spline.getKnot(i).transpose() << " q: "
                << so3_spline.getKnot(i).unit_quaternion().coeffs().transpose()
                << std::endl;
    }
  }

  inline void print_pos_knots() const {
    for (size_t i = 0; i < pos_spline.getKnots().size(); i++) {
      std::cerr << pos_spline.getKnot(i).transpose() << std::endl;
    }
  }

  inline int64_t getDtNs() const { return dt_ns; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  RdSpline<3, _N, _Scalar> pos_spline;
  So3Spline<_N, _Scalar> so3_spline;

  int64_t dt_ns;
};

}  // namespace basalt
