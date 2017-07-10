
#ifndef _PRECISION_PARAMETER_
#define _PRECISION_PARAMETER_

#include <Eigen/Core>
#include <random>


class PrecisionParameter
{
private:
  double _data;
  double shapePrior;
  double ratePrior;

public:
  PrecisionParameter(
		     const double &Value,
		     const double &shape = 0.0,
		     const double &rate = 0.0
		     ) :
    _data(Value), shapePrior(shape), ratePrior(rate)
  { ; }

  operator double () const {
    return (_data);
  };

  void update(std::mt19937 &engine, const double &rss, const int &n) {
    double shape = shapePrior + n / 2.0;
    double rate = ratePrior + rss / 2.0;
    std::gamma_distribution< double > Gamma(shape, 1 / rate);
    _data = Gamma(engine);
  };

};

#endif  // _PRECISION_PARAMETER_

