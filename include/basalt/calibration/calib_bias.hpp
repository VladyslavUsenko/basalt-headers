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
@brief Definition of static IMU biases used for calibration
*/

#pragma once

#include <Eigen/Dense>

namespace basalt {

template <typename Scalar>
class CalibAccelBias {
 public:
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Mat33 = Eigen::Matrix<Scalar, 3, 3>;

  inline CalibAccelBias() { accel_bias_full.setZero(); }

  inline void setRandom() {
    accel_bias_full.setRandom();
    accel_bias_full.template head<3>() /= 10;
    accel_bias_full.template tail<6>() /= 100;
  }

  inline const Eigen::Matrix<Scalar, 9, 1>& getParam() const {
    return accel_bias_full;
  }

  inline Eigen::Matrix<Scalar, 9, 1>& getParam() { return accel_bias_full; }

  inline void operator+=(const Eigen::Matrix<Scalar, 9, 1>& inc) {
    accel_bias_full += inc;
  }

  inline void getBiasAndScale(Vec3& accel_bias, Mat33& accel_scale) const {
    accel_bias = accel_bias_full.template head<3>();

    accel_scale.setZero();
    accel_scale.col(0) = accel_bias_full.template segment<3>(3);
    accel_scale(1, 1) = accel_bias_full(6);
    accel_scale(2, 1) = accel_bias_full(7);
    accel_scale(2, 2) = accel_bias_full(8);
  }

  inline Vec3 getCalibrated(const Vec3& raw_measurements) const {
    Vec3 accel_bias;
    Mat33 accel_scale;

    getBiasAndScale(accel_bias, accel_scale);

    return (raw_measurements + accel_scale * raw_measurements - accel_bias);
  }

  inline Vec3 invertCalibration(const Vec3& calibrated_measurements) const {
    Vec3 accel_bias;
    Mat33 accel_scale;

    getBiasAndScale(accel_bias, accel_scale);

    Mat33 accel_scale_inv =
        (Eigen::Matrix3d::Identity() + accel_scale).inverse();

    return accel_scale_inv * (calibrated_measurements + accel_bias);
  }

 private:
  Eigen::Matrix<Scalar, 9, 1> accel_bias_full;
};

template <typename Scalar>
class CalibGyroBias {
 public:
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Mat33 = Eigen::Matrix<Scalar, 3, 3>;

  inline CalibGyroBias() { gyro_bias_full.setZero(); }

  inline void setRandom() {
    gyro_bias_full.setRandom();
    gyro_bias_full.template head<3>() /= 10;
    gyro_bias_full.template tail<9>() /= 100;
  }

  inline const Eigen::Matrix<Scalar, 12, 1>& getParam() const {
    return gyro_bias_full;
  }

  inline Eigen::Matrix<Scalar, 12, 1>& getParam() { return gyro_bias_full; }

  inline void operator+=(const Eigen::Matrix<Scalar, 12, 1>& inc) {
    gyro_bias_full += inc;
  }

  inline void getBiasAndScale(Vec3& gyro_bias, Mat33& gyro_scale) const {
    gyro_bias = gyro_bias_full.template head<3>();
    gyro_scale.col(0) = gyro_bias_full.template segment<3>(3);
    gyro_scale.col(1) = gyro_bias_full.template segment<3>(6);
    gyro_scale.col(2) = gyro_bias_full.template segment<3>(9);
  }

  inline Vec3 getCalibrated(const Vec3& raw_measurements) const {
    Vec3 gyro_bias;
    Mat33 gyro_scale;

    getBiasAndScale(gyro_bias, gyro_scale);

    return (raw_measurements + gyro_scale * raw_measurements - gyro_bias);
  }

  inline Vec3 invertCalibration(const Vec3& calibrated_measurements) const {
    Vec3 gyro_bias;
    Mat33 gyro_scale;

    getBiasAndScale(gyro_bias, gyro_scale);

    Mat33 gyro_scale_inv = (Eigen::Matrix3d::Identity() + gyro_scale).inverse();

    return gyro_scale_inv * (calibrated_measurements + gyro_bias);
  }

 private:
  Eigen::Matrix<Scalar, 12, 1> gyro_bias_full;
};

}  // namespace basalt
