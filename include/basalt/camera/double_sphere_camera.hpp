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

#include <Eigen/Dense>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

template <typename Scalar = double>
class DoubleSphereCamera {
 public:
  static constexpr int N = 6;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  DoubleSphereCamera() { param.setZero(); }

  explicit DoubleSphereCamera(const VecN& p) { param = p; }

  template <class Scalar2>
  DoubleSphereCamera<Scalar2> cast() const {
    return DoubleSphereCamera<Scalar2>(param.template cast<Scalar2>());
  }

  static std::string getName() { return "ds"; }

  inline bool project(const Vec4& p3d, Vec2& proj,
                      Mat24* d_proj_d_p3d = nullptr,
                      Mat2N* d_proj_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar& x = p3d[0];
    const Scalar& y = p3d[1];
    const Scalar& z = p3d[2];

    const Scalar xx = x * x;
    const Scalar yy = y * y;
    const Scalar zz = z * z;

    if (xx + yy + zz < Sophus::Constants<Scalar>::epsilon()) return false;

    const Scalar r2 = xx + yy;

    const Scalar d1_2 = r2 + zz;
    const Scalar d1 = std::sqrt(d1_2);

    const Scalar k = xi * d1 + z;
    const Scalar kk = k * k;

    const Scalar d2_2 = r2 + kk;
    const Scalar d2 = std::sqrt(d2_2);

    const Scalar norm = alpha * d2 + (1 - alpha) * k;
    const Scalar norm_inv = Scalar(1.0) / norm;

    const Scalar mx = x * norm_inv;
    const Scalar my = y * norm_inv;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;

    if (d_proj_d_p3d || d_proj_d_param) {
      const Scalar d2_inv = Scalar(1.0) / d2;
      const Scalar norm_inv2 = norm_inv * norm_inv;

      if (d_proj_d_p3d) {
        const Scalar d1_inv = Scalar(1.0) / d1;
        const Scalar xy = x * y;
        const Scalar tt2 = xi * z * d1_inv + 1;

        const Scalar d_norm_d_r2 = (xi * (1 - alpha) * d1_inv +
                                    alpha * (xi * k * d1_inv + 1) * d2_inv) *
                                   norm_inv2;

        const Scalar tmp2 =
            ((1 - alpha) * tt2 + alpha * k * tt2 * d2_inv) * norm_inv2;

        (*d_proj_d_p3d)(0, 0) = fx * (norm_inv - xx * d_norm_d_r2);
        (*d_proj_d_p3d)(1, 0) = -fy * xy * d_norm_d_r2;

        (*d_proj_d_p3d)(0, 1) = -fx * xy * d_norm_d_r2;
        (*d_proj_d_p3d)(1, 1) = fy * (norm_inv - yy * d_norm_d_r2);

        (*d_proj_d_p3d)(0, 2) = -fx * x * tmp2;
        (*d_proj_d_p3d)(1, 2) = -fy * y * tmp2;

        (*d_proj_d_p3d)(0, 3) = 0;
        (*d_proj_d_p3d)(1, 3) = 0;
      }

      if (d_proj_d_param) {
        (*d_proj_d_param).setZero();
        (*d_proj_d_param)(0, 0) = mx;
        (*d_proj_d_param)(0, 2) = 1;
        (*d_proj_d_param)(1, 1) = my;
        (*d_proj_d_param)(1, 3) = 1;

        const Scalar tmp4 = (alpha - 1 - alpha * k * d2_inv) * d1 * norm_inv2;
        const Scalar tmp5 = (k - d2) * norm_inv2;

        (*d_proj_d_param)(0, 4) = fx * x * tmp4;
        (*d_proj_d_param)(1, 4) = fy * y * tmp4;

        (*d_proj_d_param)(0, 5) = fx * x * tmp5;
        (*d_proj_d_param)(1, 5) = fy * y * tmp5;
      }
    }

    return true;
  }

  inline bool unproject(const Vec2& proj, Vec4& p3d,
                        Mat42* d_p3d_d_proj = nullptr,
                        Mat4N* d_p3d_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar mx = (proj[0] - cx) / fx;
    const Scalar my = (proj[1] - cy) / fy;

    const Scalar r2 = mx * mx + my * my;

    const Scalar xi2_2 = alpha * alpha;
    const Scalar xi1_2 = xi * xi;

    const Scalar sqrt2 = std::sqrt(1 - (2 * alpha - 1) * r2);

    const Scalar norm2 = alpha * sqrt2 + 1 - alpha;
    const Scalar norm2_inv = Scalar(1.0) / norm2;

    const Scalar mz = (1 - xi2_2 * r2) * norm2_inv;
    const Scalar mz2 = mz * mz;

    const Scalar norm1 = mz2 + r2;
    const Scalar norm1_inv = Scalar(1.0) / norm1;
    const Scalar sqrt1 = std::sqrt(mz2 + (1 - xi1_2) * r2);
    const Scalar k = (mz * xi + sqrt1) * norm1_inv;

    p3d[0] = k * mx;
    p3d[1] = k * my;
    p3d[2] = k * mz - xi;
    p3d[3] = 0;

    if (d_p3d_d_proj || d_p3d_d_param) {
      const Scalar sqrt1_inv = Scalar(1.0) / sqrt1;
      const Scalar sqrt2_inv = Scalar(1.0) / sqrt2;
      const Scalar norm2_inv2 = norm2_inv * norm2_inv;
      const Scalar norm1_inv2 = norm1_inv * norm1_inv;

      const Scalar d_mz_d_r2 =
          (0.5 * alpha - xi2_2) * (r2 * xi2_2 - 1) * sqrt2_inv * norm2_inv2 -
          xi2_2 * norm2_inv;

      const Scalar d_mz_d_mx = 2 * mx * d_mz_d_r2;
      const Scalar d_mz_d_my = 2 * my * d_mz_d_r2;

      const Scalar d_k_d_mz =
          (norm1 * (xi * sqrt1 + mz) - 2 * mz * (mz * xi + sqrt1) * sqrt1) *
          norm1_inv2 * sqrt1_inv;

      const Scalar d_k_d_r2 =
          (xi * d_mz_d_r2 +
           0.5 * sqrt1_inv * (2 * mz * d_mz_d_r2 + 1 - xi1_2)) *
              norm1_inv -
          (mz * xi + sqrt1) * (2 * mz * d_mz_d_r2 + 1) * norm1_inv2;

      const Scalar d_k_d_mx = d_k_d_r2 * 2 * mx;
      const Scalar d_k_d_my = d_k_d_r2 * 2 * my;

      const Scalar fx_inv = Scalar(1.0) / fx;
      const Scalar fy_inv = Scalar(1.0) / fy;

      Vec4 c0, c1;

      c0[0] = fx_inv * (mx * d_k_d_mx + k);
      c0[1] = fx_inv * my * d_k_d_mx;
      c0[2] = fx_inv * (mz * d_k_d_mx + k * d_mz_d_mx);
      c0[3] = 0;

      c1[0] = fy_inv * mx * d_k_d_my;
      c1[1] = fy_inv * (my * d_k_d_my + k);
      c1[2] = fy_inv * (mz * d_k_d_my + k * d_mz_d_my);
      c1[3] = 0;

      if (d_p3d_d_proj) {
        d_p3d_d_proj->col(0) = c0;
        d_p3d_d_proj->col(1) = c1;
      }

      if (d_p3d_d_param) {
        const Scalar d_k_d_xi1 = (mz * sqrt1 - xi * r2) * sqrt1_inv * norm1_inv;

        const Scalar d_mz_d_xi2 = (1 - r2 * xi2_2) *
                                      (r2 * alpha * sqrt2_inv - sqrt2 + 1) *
                                      norm2_inv2 -
                                  2 * r2 * alpha * norm2_inv;

        const Scalar d_k_d_xi2 = d_k_d_mz * d_mz_d_xi2;

        (*d_p3d_d_param).col(0) = -c0 * mx;
        (*d_p3d_d_param).col(1) = -c1 * my;

        (*d_p3d_d_param).col(2) = -c0;
        (*d_p3d_d_param).col(3) = -c1;

        (*d_p3d_d_param)(0, 4) = mx * d_k_d_xi1;
        (*d_p3d_d_param)(1, 4) = my * d_k_d_xi1;
        (*d_p3d_d_param)(2, 4) = mz * d_k_d_xi1 - 1;
        (*d_p3d_d_param)(3, 4) = 0;

        (*d_p3d_d_param)(0, 5) = mx * d_k_d_xi2;
        (*d_p3d_d_param)(1, 5) = my * d_k_d_xi2;
        (*d_p3d_d_param)(2, 5) = mz * d_k_d_xi2 + k * d_mz_d_xi2;
        (*d_p3d_d_param)(3, 5) = 0;
      }
    }

    return true;
  }

  inline void setFromInit(const Vec4& init) {
    param[0] = init[0];
    param[1] = init[1];
    param[2] = init[2];
    param[3] = init[3];
    param[4] = 0;
    param[5] = 0.5;
  }

  void operator+=(const VecN& inc) { param += inc; }

  const VecN& getParam() const { return param; }

  static Eigen::vector<DoubleSphereCamera> getTestProjections() {
    Eigen::vector<DoubleSphereCamera> res;

    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785;
    res.emplace_back(vec1);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

}  // namespace basalt
