//back-tracking for filter smoother

#ifndef BACKTRACK_H
#define BACKTRACK_H

#include <RcppArmadillo.h>

void backtrack_pf(arma::cube& alpha, const arma::umat& ind);

#endif
