
#ifndef _GET_UNIVARIATE_COEFS_
#define _GET_UNIVARIATE_COEFS_

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

Eigen::MatrixXd getUnivariateCoefs(
				   const Eigen::Map< Eigen::VectorXd > &y,
				   const Eigen::Map< Eigen::MatrixXd > &X
) {
  Eigen::MatrixXd coefs(X.cols(), 2);  // store estimates and SE's
  Eigen::MatrixXd Z = Eigen::MatrixXd::Ones(y.size(), 2);
  Eigen::Matrix2d A;  // (Z^T Z)^-1
  Eigen::LLT < Eigen::Matrix2d > llt;
  Eigen::Vector2d betaHat;
  Eigen::VectorXd residuals(y.size());
  double sigmaSq;
  double M = 0.0; // max coefficient
  for (int j = 1; j < X.cols(); j++) {
    Z.col(1) = X.col(j);
    llt.compute(Z.transpose() * Z);
    betaHat = llt.solve(Z.transpose() * y);
    residuals = y - Z * betaHat;
    sigmaSq = residuals.transpose() * residuals;
    sigmaSq /= (y.size() - Z.cols());
    A = llt.matrixL();
    A = A.inverse();
    A = A.transpose() * A;
    coefs(j, 0) = betaHat(1);
    coefs(j, 1) = std::sqrt(sigmaSq * A(1, 1));
    M = std::max(M, std::abs(betaHat(1)));
    if (M == std::abs(betaHat(1))) {
      coefs(0, 0) = betaHat(0);
      coefs(0, 1) = std::sqrt(sigmaSq * A(0, 0));
    }
  }
  return (coefs);
};

#endif  // _GET_UNIVARIATE_COEFS_
