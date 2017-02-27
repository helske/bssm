//back-tracking for filter smoother

#ifndef FILTERSMOOTHER_H
#define FILTERSMOOTHER_H

#include <RcppArmadillo.h>

void filter_smoother(arma::cube& alpha, const arma::umat& indices);

#endif
