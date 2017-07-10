
#ifndef _THRESH_COEF_
#define _THRESH_COEF_

#include <random>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "RandWalkParam.h"
#include "sqrt_Eigen.h"
#include "scaleKernel.h"


class ThreshCoef: public Eigen::VectorXd
{
private:
  double muPrior;
  double tauPrior;
  RandWalkParam< Eigen::ArrayXd > walk;

  double logPosterior(
		      const Eigen::VectorXd &theta,
		      const Eigen::Map< Eigen::VectorXd > &y,
		      const Eigen::Map< Eigen::MatrixXd > &X,
		      const double &lambda,
		      const double &tauSq
		      ) const {
    Eigen::SparseVector< double > beta = theta.sparseView(1, lambda);
    beta.insert(0) = theta(0);
    Eigen::VectorXd res = y - X * beta;
    Eigen::VectorXd ctheta = theta - VectorXd::Ones(theta.size()) * muPrior;
    double rss = res.transpose() * res;
    double prior = ctheta.transpose() * ctheta;
    return (-0.5 * (tauSq * rss + tauPrior * prior));
  };

  double logPosterior(
		      const Eigen::VectorXd &theta,
		      const double &tauSq,
		      const double &residSS
		      ) const {
    Eigen::VectorXd ctheta = theta - VectorXd::Ones(theta.size()) * muPrior;
    double prior = ctheta.transpose() * ctheta;
    return (-0.5 * (tauSq * residSS + tauPrior * prior));
  };


public:
  ThreshCoef() : Eigen::VectorXd()
  { ; }

  // This constructor allows ThreshCoef construction from Eigen expressions
  template < typename OtherDerived >
  ThreshCoef(const Eigen::MatrixBase< OtherDerived > &other) :
    Eigen::VectorXd(other)
  { ; }

  // Overloaded = allows Eigen expressions to be assigned to a ThreshCoef object
  template < typename OtherDerived >
  ThreshCoef& operator = (const Eigen::MatrixBase< OtherDerived > &other) {
    this->Eigen::VectorXd::operator=(other);
    return (*this);
  };

  Eigen::VectorXd getKernel() const {
    return (walk.getKernel().matrix());
  };

  double getJumpProb() const {
    return (walk.getProb());
  };

  void setKernel(const Eigen::VectorXd &Kern) {
    if (Kern.minCoeff() <= 0)
      throw (std::logic_error("ThreshCoef jumping kernel <= 0"));
    walk.setKernel(Kern.array());
    walk.setScale(2.4 / std::sqrt(Kern.size()));
  };

  void setKernelScale(const double &scale) {
    if (scale <= 0)
      throw (std::logic_error("ThreshCoef kernel scale factor must be > 0"));
    walk.setScale(scale);
  }

  void setPriors(const double &mu, const double &tau) {
    if (tau <= 0)
      throw (std::logic_error("ThreshCoef precision parameter must be > 0"));
    muPrior = mu;
    tauPrior = tau;
  };

  void update(
		std::mt19937 &engine,
		const Eigen::Map< Eigen::VectorXd > &y,
		const Eigen::Map< Eigen::MatrixXd > &X,
		const double &lambda,
		const double &tauSq,
		const double &residSS
		) {
    static std::normal_distribution< double > z(0.0, 1.0);
    static std::uniform_real_distribution< double > unif(0.0, 1.0);
    Eigen::VectorXd proposal = *this;
    Eigen::VectorXd kernel = walk.getKernel().matrix();
    double p = 0.0, r;
    for (int i = 0; i < this->size(); i++)
      proposal(i) += kernel(i) * z(engine);
    r = std::exp(logPosterior(proposal, y, X, lambda, tauSq) -
		 logPosterior(*this, tauSq, residSS)
		 );
    p = std::min(r, 1.0);
    if (unif(engine) < r)
      *this = proposal;
    walk.updateParams(this->array(), p);
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

  Eigen::SparseVector< double > getSparseCoefs(const double &lambda) const {
    Eigen::SparseVector< double > beta = this->sparseView(1, lambda);
    beta.insert(0) = (*this)(0);
    return (beta);
  };

};

#endif  // _THRESH_COEF_
