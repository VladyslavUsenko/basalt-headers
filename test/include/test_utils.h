#include "gtest/gtest.h"

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

template <typename Derived1, typename Derived2, typename F>
void test_jacobian(const std::string &name,
                   const Eigen::MatrixBase<Derived1> &Ja, F func,
                   const Eigen::MatrixBase<Derived2> &x0, double eps = 1e-8,
                   double max_norm = 1e-3) {
  typedef typename Derived1::Scalar Scalar;

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Jn = Ja;
  Jn.setZero();

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> inc = x0;
  for (int i = 0; i < Jn.cols(); i++) {
    inc.setZero();
    inc[i] += eps;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fpe = func(x0 + inc);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> fme = func(x0 - inc);

    Jn.col(i) = (fpe - fme) / (2 * eps);
  }

  EXPECT_TRUE(Ja.allFinite()) << name << ": Ja not finite\n " << Ja;
  EXPECT_TRUE(Jn.allFinite()) << name << ": Jn not finite\n " << Jn;

  if (Jn.isZero(max_norm) && Ja.isZero(max_norm)) {
    EXPECT_TRUE((Jn - Ja).isZero(max_norm))
        << name << ": Ja not equal to Jn(diff norm:" << (Jn - Ja).norm()
        << ")\nJa: (norm: " << Ja.norm() << ")\n"
        << Ja << "\nJn: (norm: " << Jn.norm() << ")\n"
        << Jn;
    //<< "\ndiff:\n" << Jn - Ja;
  } else {
    EXPECT_TRUE(Jn.isApprox(Ja, max_norm))
        << name << ": Ja not equal to Jn (diff norm:" << (Jn - Ja).norm()
        << ")\nJa: (norm: " << Ja.norm() << ")\n"
        << Ja << "\nJn: (norm: " << Jn.norm() << ")\n"
        << Jn;
    //<< "\ndiff:\n" << Jn - Ja;
  }
}

#endif
