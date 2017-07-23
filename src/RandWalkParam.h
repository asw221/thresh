
#ifndef _RAND_WALK_PARAM_
#define _RAND_WALK_PARAM_

#include <algorithm>
#include <cmath>

using namespace std;


template < typename T = double >
class RandWalkParam
{
private:
  T Kernel;        // jumping kernel
  T firstMoment;   // var's approx 1st moment for updating kernel
  T secondMoment;  // var's approx 2nd moment for updating kernel
  double scale;    // kernel scaling factor
  double p;        // jumping probability
  int n;           // number of iterations since last update

public:
  RandWalkParam() {
    p = 0.0;
    n = 0;
    scale = 1.0;
  };

  RandWalkParam(const T &Kern) {
    scale = 1.0;
    setKernel(Kern);
  };


  // Sets the data used to update the jumping kernel to 0. Does NOT
  // zero the kernel itself nor the kernel scaling constant
  void clear() {
    p = 0.0;
    n = 0;
    firstMoment *= 0;
    secondMoment *= 0;
  };

  void updateParams(const T &data, const double &prob) {
    n++;
    p += prob;
    firstMoment += data;
    secondMoment += data * data;
  };

  void updateKernel() {
    T fm = getFirstMoment();
    Kernel = sqrt(getSecondMoment() - (fm * fm));
  };

  void setScale(const double &sc) {
    scale = sc;
  };

  void setKernel(const T &Kern) {
    Kernel = Kern;
    firstMoment = Kern;
    secondMoment = Kern;
    clear();
  };

  double getProb() const {
    return (p / max(n, 1));
  };

  int getN() const {
    return (n);
  };

  T getKernel() const {
    return (Kernel * scale);
  };


  T getFirstMoment() const {
    return (firstMoment / max(n, 1));
  };

  T getSecondMoment() const {
    return (secondMoment / max(n - 1, 1));
  };

  double getScale() const {
    return (scale);
  };

  // void print () const {
  //   cout << "Kernel:" << std::endl << getKernel() << std::endl
  //        << "First Moment:" << std::endl << getFirstMoment() << std::endl
  //        << "Second Moment:" << std::endl << getSecondMoment() << std::endl
  //        << "scale:" << std::endl << getScale() << std::endl
  //        << "p:" << std::endl << getProb() << std::endl
  //        << "N:" << std::endl << getN() << std::endl << std::endl;
  // };

};

#endif  // _RAND_WALK_PARAM_
