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

#include <basalt/utils/sophus_utils.hpp>
#include <memory>

namespace basalt {

constexpr size_t POSE_SIZE = 6;
constexpr size_t POSE_VEL_SIZE = 9;
constexpr size_t POSE_VEL_BIAS_SIZE = 15;

struct PoseState {
  using VecN = Eigen::Matrix<double, POSE_SIZE, 1>;

  PoseState() { t_ns = 0; }

  PoseState(int64_t t_ns, const Sophus::SE3d& T_w_i)
      : t_ns(t_ns), T_w_i(T_w_i) {}

  void applyInc(const VecN& inc) { incPose(inc, T_w_i); }

  inline static void incPose(const Sophus::Vector6d& inc, Sophus::SE3d& T) {
    T.translation() += inc.head<3>();
    T.so3() = Sophus::SO3d::exp(inc.tail<3>()) * T.so3();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int64_t t_ns;
  Sophus::SE3d T_w_i;
};

struct PoseVelState : public PoseState {
  using VecN = Eigen::Matrix<double, POSE_VEL_SIZE, 1>;

  PoseVelState() { vel_w_i.setZero(); };

  PoseVelState(int64_t t_ns, const Sophus::SE3d& T_w_i,
               const Eigen::Vector3d& vel_w_i)
      : PoseState(t_ns, T_w_i), vel_w_i(vel_w_i) {}

  void applyInc(const VecN& inc) {
    PoseState::applyInc(inc.head<6>());
    vel_w_i += inc.tail<3>();
  }

  VecN diff(const PoseVelState& other) const {
    VecN res;
    res.segment<3>(0) = other.T_w_i.translation() - T_w_i.translation();
    res.segment<3>(3) = (other.T_w_i.so3() * T_w_i.so3().inverse()).log();
    res.tail<3>() = other.vel_w_i - vel_w_i;
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d vel_w_i;
};

struct PoseVelBiasState : public PoseVelState {
  using Ptr = std::shared_ptr<PoseVelBiasState>;
  using VecN = Eigen::Matrix<double, POSE_VEL_BIAS_SIZE, 1>;

  PoseVelBiasState() {
    bias_gyro.setZero();
    bias_accel.setZero();
  };

  PoseVelBiasState(int64_t t_ns, const Sophus::SE3d& T_w_i,
                   const Eigen::Vector3d& vel_w_i,
                   const Eigen::Vector3d& bias_gyro,
                   const Eigen::Vector3d& bias_accel)
      : PoseVelState(t_ns, T_w_i, vel_w_i),
        bias_gyro(bias_gyro),
        bias_accel(bias_accel) {}

  void applyInc(const VecN& inc) {
    PoseVelState::applyInc(inc.head<9>());
    bias_gyro += inc.segment<3>(9);
    bias_accel += inc.segment<3>(12);
  }

  VecN diff(const PoseVelBiasState& other) const {
    VecN res;
    res.segment<9>(0) = PoseVelState::diff(other);
    res.segment<3>(9) = other.bias_gyro - bias_gyro;
    res.segment<3>(12) = other.bias_accel - bias_accel;
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d bias_gyro;
  Eigen::Vector3d bias_accel;
};

struct ImuData {
  using Ptr = std::shared_ptr<ImuData>;

  int64_t t_ns;
  Eigen::Vector3d accel;
  Eigen::Vector3d gyro;

  Eigen::Vector3d accel_cov;
  Eigen::Vector3d gyro_cov;

  ImuData() {
    accel.setZero();
    gyro.setZero();

    accel_cov.setZero();
    gyro_cov.setZero();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace basalt
