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

#include <basalt/utils/sophus_utils.hpp>

#include <sophus/se2.hpp>
#include <sophus/sim2.hpp>

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(SophusUtilsCase, RightJacobianSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d Ja, Jn;
  Sophus::rightJacobianSO3(phi, Ja);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianSO3", Ja,
      [&](const Eigen::Vector3d &x) {
        return (Sophus::SO3d::exp(phi).inverse() * Sophus::SO3d::exp(phi + x))
            .log();
      },
      x0);
}

TEST(SophusUtilsCase, RightJacobianInvSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d Ja, Jn;
  Sophus::rightJacobianInvSO3(phi, Ja);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianInvSO3", Ja,
      [&](const Eigen::Vector3d &x) {
        return (Sophus::SO3d::exp(phi) * Sophus::SO3d::exp(x)).log();
      },
      x0);
}

TEST(SophusUtilsCase, LeftJacobianSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d Ja, Jn;
  Sophus::leftJacobianSO3(phi, Ja);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "leftJacobianSO3", Ja,
      [&](const Eigen::Vector3d &x) {
        return (Sophus::SO3d::exp(phi + x) * Sophus::SO3d::exp(phi).inverse())
            .log();
      },
      x0);
}

TEST(SophusUtilsCase, LeftJacobianInvSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d Ja, Jn;
  Sophus::leftJacobianInvSO3(phi, Ja);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "leftJacobianInvSO3", Ja,
      [&](const Eigen::Vector3d &x) {
        return (Sophus::SO3d::exp(x) * Sophus::SO3d::exp(phi)).log();
      },
      x0);
}

TEST(SophusUtilsCase, RightJacobianSE3Decoupled) {
  Sophus::Vector6d phi;
  phi.setRandom();

  Sophus::Matrix6d Ja, Jn;
  Sophus::rightJacobianSE3Decoupled(phi, Ja);

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianSE3Decoupled", Ja,
      [&](const Sophus::Vector6d &x) {
        return Sophus::se3_logd(Sophus::se3_expd(phi).inverse() *
                                Sophus::se3_expd(phi + x));
      },
      x0);
}

TEST(SophusUtilsCase, RightJacobianInvSE3Decoupled) {
  Sophus::Vector6d phi;
  phi.setRandom();

  Sophus::Matrix6d Ja, Jn;
  Sophus::rightJacobianInvSE3Decoupled(phi, Ja);

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianInvSE3Decoupled", Ja,
      [&](const Sophus::Vector6d &x) {
        return Sophus::se3_logd(Sophus::se3_expd(phi) * Sophus::se3_expd(x));
      },
      x0);
}

TEST(SophusUtilsCase, Adjoint) {
  Sophus::Vector6d phi;
  phi.setRandom();

  Sophus::SE3d pose = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  Sophus::Matrix6d Ja = pose.inverse().Adj();

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "Adj", Ja,
      [&](const Sophus::Vector6d &x) {
        return Sophus::se3_logd(pose.inverse() * Sophus::se3_expd(x) * pose);
      },
      x0);
}

TEST(SophusUtilsCase, RotTestSO3) {
  Eigen::Vector3d t1 = Eigen::Vector3d::Random();
  Eigen::Vector3d t2 = Eigen::Vector3d::Random();

  double k = 0.6234234;

  Eigen::Matrix3d J;
  J.setZero();

  Sophus::rightJacobianSO3(k * t1, J);
  J = -k * Sophus::SO3d::exp(k * t1).matrix() * Sophus::SO3d::hat(t2) * J;

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "Rot Test", J,
      [&](const Eigen::Vector3d &x) {
        return Sophus::SO3d::exp(k * (t1 + x)) * t2;
      },
      x0);
}

TEST(SophusUtilsCase, incTest) {
  Sophus::Vector6d phi;
  phi.setRandom();

  Sophus::SE3d pose = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  Sophus::Matrix6d Ja;
  Ja.setZero();

  Eigen::Matrix3d R = pose.so3().inverse().matrix();
  Ja.topLeftCorner<3, 3>() = R;
  Ja.bottomRightCorner<3, 3>() = R;

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "inc test", Ja,
      [&](const Sophus::Vector6d &x) {
        Sophus::SE3d pose1;
        pose1.so3() = Sophus::SO3d::exp(x.tail<3>()) * pose.so3();
        pose1.translation() = pose.translation() + x.head<3>();

        return Sophus::se3_logd(pose.inverse() * pose1);
      },
      x0);
}

TEST(SophusUtilsCase, SO2Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 1> Ja;
  Ja[0] = -phi[1];
  Ja[1] = phi[0];

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", Ja,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        return Sophus::SO2d::exp(x[0]) * phi;
      },
      x0);
}

TEST(SophusUtilsCase, Se3Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 3> Ja;
  Ja.topLeftCorner<2, 2>().setIdentity();
  Ja(0, 2) = -phi[1];
  Ja(1, 2) = phi[0];

  Eigen::Matrix<double, 3, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", Ja,
      [&](const Eigen::Matrix<double, 3, 1> &x) {
        return Sophus::SE2d::exp(x) * phi;
      },
      x0);
}

TEST(SophusUtilsCase, Sim2Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 4> Ja;
  Ja.topLeftCorner<2, 2>().setIdentity();
  Ja(0, 2) = -phi[1];
  Ja(1, 2) = phi[0];
  Ja(0, 3) = phi[0];
  Ja(1, 3) = phi[1];

  Eigen::Matrix<double, 4, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", Ja,
      [&](const Eigen::Matrix<double, 4, 1> &x) {
        return Sophus::Sim2d::exp(x) * phi;
      },
      x0);
}

TEST(SophusUtilsCase, RxSO2Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 2> Ja;
  Ja(0, 0) = -phi[1];
  Ja(1, 0) = phi[0];
  Ja(0, 1) = phi[0];
  Ja(1, 1) = phi[1];

  Eigen::Matrix<double, 2, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", Ja,
      [&](const Eigen::Matrix<double, 2, 1> &x) {
        return Sophus::RxSO2d::exp(x) * phi;
      },
      x0);
}
