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

#include <memory>

#include <basalt/spline/rd_spline.h>
#include <basalt/calibration/calib_bias.hpp>
#include <basalt/camera/generic_camera.hpp>

namespace basalt {

template <class Scalar>
struct Calibration {
  using Ptr = std::shared_ptr<Calibration>;
  using SE3 = Sophus::SE3<Scalar>;

  Calibration() {
    cam_time_offset_ns = 0;

    // reasonable defaults
    gyro_noise_std = 0.004;
    accel_noise_std = 0.23;
    accel_bias_std = 0.001;
    gyro_bias_std = 0.0001;
  }

  template <class Scalar2>
  Calibration<Scalar2> cast() const {
    Calibration<Scalar2> new_cam;

    for (const auto& v : T_i_c)
      new_cam.T_i_c.emplace_back(v.template cast<Scalar2>());
    for (const auto& v : intrinsics)
      new_cam.intrinsics.emplace_back(v.template cast<Scalar2>());
    for (const auto& v : vignette)
      new_cam.vignette.emplace_back(v.template cast<Scalar2>());

    new_cam.resolution = resolution;
    new_cam.cam_time_offset_ns = cam_time_offset_ns;

    new_cam.calib_accel_bias.getParam() =
        calib_accel_bias.getParam().template cast<Scalar2>();
    new_cam.calib_gyro_bias.getParam() =
        calib_gyro_bias.getParam().template cast<Scalar2>();

    new_cam.gyro_noise_std = gyro_noise_std;
    new_cam.accel_noise_std = accel_noise_std;
    new_cam.gyro_bias_std = gyro_bias_std;
    new_cam.accel_bias_std = accel_bias_std;

    return new_cam;
  }

  // transfomrations from cameras to IMU
  Eigen::vector<SE3> T_i_c;

  // Camera intrinsics
  Eigen::vector<GenericCamera<Scalar>> intrinsics;

  // Camera resolutions
  Eigen::vector<Eigen::Vector2i> resolution;

  // Spline representing radially symmetric vignetting
  std::vector<basalt::RdSpline<1, 4, Scalar>> vignette;

  int64_t cam_time_offset_ns;

  // Constant pre-calibrated bias for accel and gyro
  CalibAccelBias<Scalar> calib_accel_bias;
  CalibGyroBias<Scalar> calib_gyro_bias;

  Scalar gyro_noise_std;   // [ rad / s ]   ( gyro "white noise" )
  Scalar accel_noise_std;  // [ m / s^2 ]   ( accel "white noise" )

  Scalar gyro_bias_std;
  Scalar accel_bias_std;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <class Scalar>
struct MocapCalibration {
  using Ptr = std::shared_ptr<MocapCalibration>;
  using SE3 = Sophus::SE3<Scalar>;

  MocapCalibration() {
    mocap_time_offset_ns = 0;
    mocap_to_imu_offset_ns = 0;
  }

  SE3 T_moc_w, T_i_mark;
  int64_t mocap_time_offset_ns;

  // Offset from initial alignment
  int64_t mocap_to_imu_offset_ns;
};

}  // namespace basalt
