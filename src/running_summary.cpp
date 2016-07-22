#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
void running_summary(arma::cube& x, arma::mat& mean_x, arma::cube& cov_x, unsigned int n) {
  
  cov_x *= n;
  for(unsigned int i = 0; i < x.n_slices; i++) {
    arma::mat diff = x.slice(i) - mean_x;
    mean_x += diff / (n + i + 1);
    for (unsigned int t = 0; t < x.n_cols; t++) {
      cov_x.slice(t) += diff.col(t) * (x.slice(i).col(t) - mean_x.col(t)).t();
    }
  }
  cov_x /= (n + x.n_slices);
}