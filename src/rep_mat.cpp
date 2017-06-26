#include "rep_mat.h"

arma::mat rep_mat(const arma::mat& x, const arma::uvec& counts) {
  arma::mat new_mat(x.n_rows, arma::accu(counts));
  arma::uvec cs_counts = arma::cumsum(counts);
  for (unsigned int i = 0; i < counts.n_elem; i++) {
    new_mat.cols(cs_counts(i) - counts(i), cs_counts(i) - 1).each_col() = x.col(i);
  }
  return new_mat;
}

arma::cube rep_cube(const arma::cube& x, const arma::uvec& counts) {
  arma::cube new_cube(x.n_rows, x.n_cols, arma::accu(counts));
  arma::uvec cs_counts = arma::cumsum(counts);
  for (unsigned int i = 0; i < counts.n_elem; i++) {
    new_cube.each_slice(arma::regspace<arma::uvec>(cs_counts(i) - counts(i), cs_counts(i) - 1)) = 
      x.slice(i);
  }
  return new_cube;
}