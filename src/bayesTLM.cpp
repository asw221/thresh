
#include <random>
#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

#include "ThreshCoef.h"
#include "ThresholdParameter.h"
#include "PrecisionParameter.h"
#include "scaleKernel.h"
#include "getUnivariateCoefs.h"


// =============================================================================
// Fit Bayesian Thresholded Linear Model using random walk MCMC for sparse
// covariate selection and effect estimation. X variables should be on roughly
// the same scale before fitting the model. R interface handled with Rcpp;
// numerics handled with the Eigen linear algebra library.
//
//                                                    Andrew Whiteman, July 2017
// =============================================================================




// compute the variance of a mapped Eigen VectorXd object
double varEigen(const Map< VectorXd > &x);


// Initialize a model coefficients object given priors and data
ThreshCoef initializeAlpha(
  const Map< MatrixXd > &X,  // Design matrix, X
  const Map< VectorXd > &y,  // Outcome variable, y
  const double &muPrior,     // Prior mean of the alpha_j's
  const double &tauPrior     // Prior precision of the alpha_j's
);


// Compute residual sum of squares given estimated coefficients and data
double computeResidSS(
  const SparseMatrix< double > &beta,  // Estimated sparse coefficients
  const Map< MatrixXd > &X,            // Design matrix, X
  const Map< VectorXd > &y             // Outcome variable, y
);


// Fit model and return MCMC simulations in an Rcpp::List object
List fitBayesTLM(
  const Map< MatrixXd > &X,  // Design matrix, X
  const Map< VectorXd > &y,  // Outcome variable, y
  const double alphaTauSq,   // Prior shape parameter for tau^2
  const double betaTauSq,    // Prior rate parameter for tau^2
  const int nSims,           // Total number of MCMC simulations to run
  const int burnin,          // Number of burnin iterations to discard
  const int nSave,           // Number of MCMC simulations to save
  const int block,           // Burnin kernel-update block size
  const int seed             // RNG seed
);




// .Call'able R interface
extern "C" SEXP bayesTLM(
  const SEXP X_,
  const SEXP y_,
  const SEXP alphaTauSq_,
  const SEXP betaTauSq_,
  const SEXP nSims_,
  const SEXP burnin_,
  const SEXP nSave_,
  const SEXP block_,
  const SEXP seed_
) {
  try {
    const Map< MatrixXd > X(as< Map< MatrixXd > >(X_));
    const Map< VectorXd > y(as< Map< VectorXd > >(y_));
    const double alphaTauSq(as< double >(alphaTauSq_));
    const double betaTauSq(as< double >(betaTauSq_));
    const int nSims(as< int >(nSims_));
    const int burnin(as< int >(burnin_));
    const int nSave(as< int >(nSave_));
    const int block(as< int >(block_));
    const int seed(as< int >(seed_));

    if (y.size() != X.rows())
      throw (std::logic_error("Design dimension does not match outcome"));

    if (alphaTauSq <= 0 || betaTauSq <= 0)
      throw (std::logic_error("Precision priors must be > 0"));

    if (nSims <= burnin)
      throw (std::logic_error("Total number of simulations must be > burnin"));

    if (block > burnin)
      throw (std::logic_error("burnin parameter must be >= block size"));

    if (nSims <= 0 || burnin <= 0 || nSave <= 0 || block <= 0)
      throw (std::logic_error("All MCMC parameters must be > 0"));


    return (wrap(fitBayesTLM(X, y, alphaTauSq, betaTauSq,
			     nSims, burnin, nSave, block, seed
			     )));

  }
  catch (std::exception& _ex_) {
    forward_exception_to_r(_ex_);
  }
  catch (...) {
    ::Rf_error("C++ exception (unknown cause)");
  }

  return (R_NilValue);  // not reached
}







List fitBayesTLM(
  const Map< MatrixXd > &X,
  const Map< VectorXd > &y,
  const double alphaTauSq,
  const double betaTauSq,
  const int nSims,
  const int burnin,
  const int nSave,
  const int block,
  const int seed
) {

  std::mt19937 rengine;
  rengine.seed(seed);


  // model parameters
  const int N = y.size();
  ThreshCoef alpha = initializeAlpha(X, y, 0, 0.001);
  const double M = std::max(std::abs(alpha.minCoeff()),
			    std::abs(alpha.maxCoeff())
			    );
  alpha.setKernelScale(2.4 / std::sqrt(alpha.getSparseCoefs(M / 2).nonZeros()));
  // try to crudely approximate the number of non-zero parameters
  // to be estimated based on uniform expectation. Set initial alpha
  // kernel scale to reflect this for (often) faster convergence

  ThresholdParameter lambda(M);
  PrecisionParameter tauSq(1 / varEigen(y), alphaTauSq, betaTauSq);
  double rss;  // residual sum of squares

  // storage
  MatrixXd alphaSims(nSave, alpha.size());
  VectorXd lambdaSims(nSave), sigmaSqSims(nSave);

  // simulation parameters
  const int thin = std::max((nSims - burnin) / nSave, 1);
  int i = 0;  // loop variable

  // Run burnin iterations, updating jumping kernels every block'th
  // iteration. Note that only alpha and lambda are updated with random
  // walk methods. tau^2 is updated with full conditional Gibbs and doesn't
  // require a kernel update
  while (i < burnin) {
    for (int j = 0; j < block; j++) {
      rss = computeResidSS(alpha.getSparseCoefs(lambda), X, y);
      alpha.update(rengine, y, X, lambda, tauSq, rss);
      lambda.update(rengine, y, X, alpha, tauSq, rss);
      tauSq.update(rengine, rss, N);
    }
    alpha.updateKernel();
    lambda.updateKernel();
    i += block;
  }

  // Print average jumping probabilities after burnin iterations
  Rcout << "After burnin (" << burnin << " iterations)\nPr(jump) ="
	<< "\nalpha: " << alpha.getJumpProb()
	<< "\nlambda: " << lambda.getJumpProb() << "\n";

  // Run non-burnin iterations and retain every thin'th simulation draw
  i = 0;
  for (int j = 0; i < nSave; j++) {
    rss = computeResidSS(alpha.getSparseCoefs(lambda), X, y);
    alpha.update(rengine, y, X, lambda, tauSq, rss);
    lambda.update(rengine, y, X, alpha, tauSq, rss);
    tauSq.update(rengine, rss, N);
    if (j % thin == 0) {
      alphaSims.row(i) = alpha;
      lambdaSims(i) = lambda;
      sigmaSqSims(i) = 1 / tauSq;
      i++;
    }
  }

  List result = List::create(
    Named("alphaHat") = alphaSims.colwise().mean(),
    Named("lambdaHat") = lambdaSims.mean(),
    Named("sigmaSqHat") = sigmaSqSims.mean(),
    Named("alpha") = alphaSims,
    Named("lambda") = lambdaSims,
    Named("sigmaSq") = sigmaSqSims
  );

  return (result);
}




double varEigen(const Map< VectorXd > &x) {
  VectorXd cx = x - VectorXd::Ones(x.size()) * x.mean();
  double temp = cx.transpose() * cx;
  return (temp / std::max(int(cx.size() - 1), 1));
}


ThreshCoef initializeAlpha(
  const Map< MatrixXd > &X,
  const Map< VectorXd > &y,
  const double &muPrior,
  const double &tauPrior
) {
  // get univariate beta's and SE's
  MatrixXd start = getUnivariateCoefs(y, X);
  ThreshCoef alpha(start.col(0));  // set starting values to univar beta's
  alpha.setKernel(start.col(1));   // set starting kernel to univar SE's
  alpha.setPriors(muPrior, tauPrior);
  return (alpha);
}


double computeResidSS(
  const SparseMatrix< double > &beta,
  const Map< MatrixXd > &X,
  const Map< VectorXd > &y
) {
  VectorXd residuals = y - X * beta;
  return (residuals.transpose() * residuals);
}

