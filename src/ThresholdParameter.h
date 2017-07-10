
#ifndef _THRESHOLD_PARAMETER_
#define _THRESHOLD_PARAMETER_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cmath>
#include <random>
#include <algorithm>
#include "RandWalkParam.h"
#include "ThreshCoef.h"


class ThresholdParameter
{
private:
  double _data;
  double Min;
  double Max;
  RandWalkParam<> walk;

  double logPosterior(
		      const double &theta,
		      const Eigen::Map< Eigen::VectorXd > &y,
		      const Eigen::Map< Eigen::MatrixXd > &X,
		      const ThreshCoef &alpha,
		      const double &tauSq
		      ) const
  {
    Eigen::VectorXd res = y - X * alpha.getSparseCoefs(theta);
    double rss = res.transpose() * res;
    return (-0.5 * tauSq * rss);
  };

  double logPosterior(
		      const double &tauSq,
		      const double &residSS
		      ) const
  {
    return (-0.5 * tauSq * residSS);
  };

public:
  ThresholdParameter(
		     const double maxVal = 1.0,
		     const double minVal = 0.0
		     ) : Min(minVal), Max(maxVal)
  {
    _data = (Max - Min) / 2;               // uniform expectation
    walk.setKernel((Max - Min) / std::sqrt(12));  // uniform sigma
    // walk.setScale(2.4);
    walk.setScale(1.0);
  };

  double operator = (const double &rhs) {
    if (rhs < Min || rhs > Max)
      throw (std::logic_error("ThresholdParameter value out of bounds"));
    _data = rhs;
    return (rhs);
  };

  operator double () const {
    return (_data);
  };

  void update(
		std::mt19937 &engine,
		const Eigen::Map< Eigen::VectorXd > &y,
		const Eigen::Map< Eigen::MatrixXd > &X,
		const ThreshCoef &alpha,
		const double &tauSq,
		const double &residSS
		) {
    static std::normal_distribution< double > z(0.0, 1.0);
    static std::uniform_real_distribution< double > unif(0.0, 1.0);
    double proposal = _data + walk.getKernel() * z(engine);
    double p = 0.0, r;
    if (proposal < Max && proposal > Min) {
      r = std::exp(logPosterior(proposal, y, X, alpha, tauSq) -
  		   logPosterior(tauSq, residSS)
		   );
      p = std::min(r, 1.0);
      if (unif(engine) < r)
	_data = proposal;
    }
    walk.updateParams(_data, p);
  };

  void updateKernel() {
    const double p = walk.getProb();
    const double newScale = scaleKernel(walk.getScale(), p);
    // if (newScale != walk.getScale() && p > 0.05 && p < 0.7)
    //   walk.updateKernel();
    if (p > 0.05 && p < 0.7)
      walk.updateKernel();
    else
      walk.clear();
    walk.setScale(newScale);
  };

  double getKernel() const {
    return (walk.getKernel());
  };

  double getJumpProb() const {
    return (walk.getProb());
  };

  void setKernel(const double &Kern) {
    if (Kern <= 0)
      throw (std::logic_error("ThresholdParameter jumping kernel <= 0"));
    walk.setKernel(Kern);
  };

};

#endif  // _THRESHOLD_PARAMETER_
