#ifndef REPMAT_H
#define REPMAT_H

#include <RcppArmadillo.h>

arma::mat rep_mat(const arma::mat& x, const arma::uvec& counts);
arma::cube rep_cube(const arma::cube& x, const arma::uvec& counts);

#endif
