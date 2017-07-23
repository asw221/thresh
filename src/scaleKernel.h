
#ifndef _SCALE_KERNEL_
#define _SCALE_KERNEL_

template < class T >
T scaleKernel(const T &Kernel, const double p) {
  double scale = ((p < 0.26) ? 0.9 : ((p > 0.34) ? 1.1 : 1));
  return (Kernel * scale);
};

#endif  // _SCALE_KERNEL_
