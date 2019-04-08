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

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

template <typename Scalar = double>
class KannalaBrandtCamera4 {
 public:
  static constexpr int N = 8;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  KannalaBrandtCamera4() { param.setZero(); }

  explicit KannalaBrandtCamera4(const VecN& p) { param = p; }

  static std::string getName() { return "kb4"; }

  template <class Scalar2>
  KannalaBrandtCamera4<Scalar2> cast() const {
    return KannalaBrandtCamera4<Scalar2>(param.template cast<Scalar2>());
  }

  inline bool project(const Vec4& p3d, Vec2& proj,
                      Mat24* d_proj_d_p3d = nullptr,
                      Mat2N* d_proj_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p3d[0];
    const Scalar& y = p3d[1];
    const Scalar& z = p3d[2];

    const Scalar r2 = x * x + y * y;
    const Scalar r = std::sqrt(r2);

    const Scalar theta = std::atan2(r, z);
    const Scalar theta2 = theta * theta;
    const Scalar theta4 = theta2 * theta2;
    const Scalar theta6 = theta4 * theta2;
    const Scalar theta8 = theta6 * theta2;

    const Scalar r_theta =
        theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

    const Scalar norm_inv = r > 1e-8 ? Scalar(1.0) / r : 1;

    const Scalar mx = r_theta * x * norm_inv;
    const Scalar my = r_theta * y * norm_inv;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;

    if (d_proj_d_p3d) {
      const Scalar z2 = z * z;

      const Scalar d_r_d_x = x * norm_inv;
      const Scalar d_r_d_y = y * norm_inv;

      const Scalar d_theta_d_arg = z2 / (z2 + r2);
      const Scalar d_theta_d_x = d_theta_d_arg * d_r_d_x / z;
      const Scalar d_theta_d_y = d_theta_d_arg * d_r_d_y / z;
      const Scalar d_theta_d_z = -r * d_theta_d_arg / z2;

      const Scalar d_r_theta_d_theta = 1 + 3 * theta2 * k1 + 5 * theta4 * k2 +
                                       7 * theta6 * k3 + 9 * theta8 * k4;

      (*d_proj_d_p3d)(0, 0) =
          fx * norm_inv * norm_inv *
          (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x -
           x * x * r_theta * norm_inv);
      (*d_proj_d_p3d)(1, 0) =
          fy * y * norm_inv * norm_inv *
          (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta * norm_inv);

      (*d_proj_d_p3d)(0, 1) =
          fx * x * norm_inv * norm_inv *
          (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta * norm_inv);

      (*d_proj_d_p3d)(1, 1) =
          fy * norm_inv * norm_inv *
          (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y -
           y * y * r_theta * norm_inv);

      (*d_proj_d_p3d)(0, 2) =
          fx * x * norm_inv * d_r_theta_d_theta * d_theta_d_z;
      (*d_proj_d_p3d)(1, 2) =
          fy * y * norm_inv * d_r_theta_d_theta * d_theta_d_z;

      (*d_proj_d_p3d)(0, 3) = 0;
      (*d_proj_d_p3d)(1, 3) = 0;
    }

    if (d_proj_d_param) {
      (*d_proj_d_param).setZero();
      (*d_proj_d_param)(0, 0) = mx;
      (*d_proj_d_param)(0, 2) = 1;
      (*d_proj_d_param)(1, 1) = my;
      (*d_proj_d_param)(1, 3) = 1;

      (*d_proj_d_param)(0, 4) = fx * x * norm_inv * theta * theta2;
      (*d_proj_d_param)(1, 4) = fy * y * norm_inv * theta * theta2;

      d_proj_d_param->col(5) = d_proj_d_param->col(4) * theta2;
      d_proj_d_param->col(6) = d_proj_d_param->col(5) * theta2;
      d_proj_d_param->col(7) = d_proj_d_param->col(6) * theta2;
    }

    return true;
  }

  template <int ITER>
  inline Scalar solve_theta(const Scalar& r_theta,
                            Scalar& d_func_d_theta) const {
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    Scalar theta = r_theta;
    for (int i = ITER; i > 0; i--) {
      Scalar theta2 = theta * theta;

      Scalar func = k4 * theta2;
      func += k3;
      func *= theta2;
      func += k2;
      func *= theta2;
      func += k1;
      func *= theta2;
      func += 1;
      func *= theta;

      d_func_d_theta = 9 * k4 * theta2;
      d_func_d_theta += 7 * k3;
      d_func_d_theta *= theta2;
      d_func_d_theta += 5 * k2;
      d_func_d_theta *= theta2;
      d_func_d_theta += 3 * k1;
      d_func_d_theta *= theta2;
      d_func_d_theta += 1;

      // Iteration of Newton method
      theta += (r_theta - func) / d_func_d_theta;
    }

    return theta;
  }

  inline bool unproject(const Vec2& proj, Vec4& p3d,
                        Mat42* d_p3d_d_proj = nullptr,
                        Mat4N* d_p3d_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar mx = (proj[0] - cx) / fx;
    const Scalar my = (proj[1] - cy) / fy;

    Scalar theta = 0, sin_theta = 0, cos_theta = 1, thetad, scaling;
    Scalar d_func_d_theta = 0;

    scaling = 1.0;
    thetad = std::sqrt(mx * mx + my * my);

    if (thetad > 1e-8) {
      theta = solve_theta<3>(thetad, d_func_d_theta);

      sin_theta = std::sin(theta);
      cos_theta = std::cos(theta);
      scaling = sin_theta / thetad;
    }

    p3d[0] = mx * scaling;
    p3d[1] = my * scaling;
    p3d[2] = cos_theta;
    p3d[3] = 0;

    if (d_p3d_d_param || d_p3d_d_param) {
      Scalar d_thetad_d_mx = 0;
      Scalar d_thetad_d_my = 0;
      Scalar d_scaling_d_thetad = 0;
      Scalar d_cos_d_thetad = 0;

      Scalar d_scaling_d_k1 = 0;
      Scalar d_cos_d_k1 = 0;

      Scalar theta2 = 0;

      if (thetad > 1e-8) {
        d_thetad_d_mx = mx / thetad;
        d_thetad_d_my = my / thetad;

        theta2 = theta * theta;

        d_scaling_d_thetad = (thetad * cos_theta / d_func_d_theta - sin_theta) /
                             (thetad * thetad);

        d_cos_d_thetad = sin_theta / d_func_d_theta;

        d_scaling_d_k1 =
            -cos_theta * theta * theta2 / (d_func_d_theta * thetad);

        d_cos_d_k1 = d_cos_d_thetad * theta * theta2;
      }

      const Scalar d_res0_d_mx =
          scaling + mx * d_scaling_d_thetad * d_thetad_d_mx;
      const Scalar d_res0_d_my = mx * d_scaling_d_thetad * d_thetad_d_my;

      const Scalar d_res1_d_mx = my * d_scaling_d_thetad * d_thetad_d_mx;
      const Scalar d_res1_d_my =
          scaling + my * d_scaling_d_thetad * d_thetad_d_my;

      const Scalar d_res2_d_mx = -d_cos_d_thetad * d_thetad_d_mx;
      const Scalar d_res2_d_my = -d_cos_d_thetad * d_thetad_d_my;

      Vec4 c0, c1;

      c0(0) = d_res0_d_mx / fx;
      c0(1) = d_res1_d_mx / fx;
      c0(2) = d_res2_d_mx / fx;
      c0(3) = 0;

      c1(0) = d_res0_d_my / fy;
      c1(1) = d_res1_d_my / fy;
      c1(2) = d_res2_d_my / fy;
      c1(3) = 0;

      if (d_p3d_d_param) {
        d_p3d_d_proj->col(0) = c0;
        d_p3d_d_proj->col(1) = c1;
      }

      if (d_p3d_d_param) {
        d_p3d_d_param->setZero();

        d_p3d_d_param->col(2) = -c0;
        d_p3d_d_param->col(3) = -c1;

        d_p3d_d_param->col(0) = -c0 * mx;
        d_p3d_d_param->col(1) = -c1 * my;

        (*d_p3d_d_param)(0, 4) = mx * d_scaling_d_k1;
        (*d_p3d_d_param)(1, 4) = my * d_scaling_d_k1;
        (*d_p3d_d_param)(2, 4) = d_cos_d_k1;
        (*d_p3d_d_param)(3, 4) = 0;

        d_p3d_d_param->col(5) = d_p3d_d_param->col(4) * theta2;
        d_p3d_d_param->col(6) = d_p3d_d_param->col(5) * theta2;
        d_p3d_d_param->col(7) = d_p3d_d_param->col(6) * theta2;
      }
    }
    return true;
  }

  void operator+=(const VecN& inc) { param += inc; }

  const VecN& getParam() const { return param; }

  inline void setFromInit(const Vec4& init) {
    param[0] = init[0];
    param[1] = init[1];
    param[2] = init[2];
    param[3] = init[3];
    param[4] = 0;
    param[5] = 0;
    param[6] = 0;
    param[7] = 0;
  }

  static Eigen::vector<KannalaBrandtCamera4> getTestProjections() {
    Eigen::vector<KannalaBrandtCamera4> res;

    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    res.emplace_back(vec1);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

}  // namespace basalt
