
#ifndef _SQRT_EIGEN_
#define _SQRT_EIGEN_

#include <Eigen/Core>

// Hack to allow for std::sqrt() like math/syntax to be carried out on
// Eigen objects. Only need for ArrayXd for now
// Andrew Whiteman - July 2017


Eigen::ArrayXd sqrt(const Eigen::ArrayXd &ary) {
  return (ary.sqrt());
}

#endif  // _SQRT_EIGEN_
