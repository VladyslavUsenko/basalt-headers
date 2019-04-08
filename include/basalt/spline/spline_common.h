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

#include <Eigen/Dense>
#include <cstdint>

namespace basalt {

inline uint64_t C_n_k(uint64_t n, uint64_t k) {
  if (k > n) {
    return 0;
  }
  uint64_t r = 1;
  for (uint64_t d = 1; d <= k; ++d) {
    r *= n--;
    r /= d;
  }
  return r;
}

template <int _N, typename _Scalar = double, bool _Cumulative = false>
Eigen::Matrix<_Scalar, _N, _N> computeBlendingMatrix() {
  Eigen::Matrix<double, _N, _N> m;
  m.setZero();

  for (int i = 0; i < _N; ++i) {
    for (int j = 0; j < _N; ++j) {
      double sum = 0;

      for (int s = j; s < _N; ++s) {
        sum += std::pow(-1.0, s - j) * C_n_k(_N, s - j) *
               std::pow(_N - s - 1.0, _N - 1.0 - i);
      }
      m(j, i) = C_n_k(_N - 1, _N - 1 - i) * sum;
    }
  }

  if (_Cumulative) {
    for (int i = 0; i < _N; i++) {
      for (int j = i + 1; j < _N; j++) {
        m.row(i) += m.row(j);
      }
    }
  }

  uint64_t factorial = 1;
  for (int i = 2; i < _N; ++i) {
    factorial *= i;
  }

  return (m / factorial).template cast<_Scalar>();
}

template <int _N, typename _Scalar = double>
Eigen::Matrix<_Scalar, _N, _N> computeBaseCoefficients() {
  Eigen::Matrix<double, _N, _N> base_coefficients;

  base_coefficients.setZero();
  base_coefficients.row(0).setOnes();

  const int DEG = _N - 1;
  int order = DEG;
  for (int n = 1; n < _N; n++) {
    for (int i = DEG - order; i < _N; i++) {
      base_coefficients(n, i) = (order - DEG + i) * base_coefficients(n - 1, i);
    }
    order--;
  }
  return base_coefficients.template cast<_Scalar>();
}

}  // namespace basalt
