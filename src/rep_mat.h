#ifndef REPMAT_H
#define REPMAT_H

#include "bssm.h"

arma::uvec rep_uvec(const arma::uvec& x, const unsigned int count);
arma::uvec rep_uvec(const arma::uvec& x, const arma::uvec& counts);
arma::vec rep_vec(const arma::vec& x, const arma::uvec& counts);
arma::mat rep_mat(const arma::mat& x, const arma::uvec& counts);
arma::cube rep_cube(const arma::cube& x, const arma::uvec& counts);

#endif
