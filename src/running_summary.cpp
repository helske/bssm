#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
void running_summary(const arma::mat& x, arma::mat& mean_x, arma::cube& cov_x, const unsigned int n) {
  
  if(n > 0){
  cov_x *= (n - 1);
  }
  
  arma::mat diff = x - mean_x;
  mean_x += diff / (n + 1);
  for (unsigned int t = 0; t < x.n_cols; t++) {
    cov_x.slice(t) += diff.col(t) * (x.col(t) - mean_x.col(t)).t();
  }
  if(n > 0){
  cov_x /= n;
  }
}

// [[Rcpp::export]]
void running_weighted_summary(const arma::cube& x, arma::mat& mean_x, arma::cube& cov_x, const arma::vec& weights) {
  
  cov_x.zeros();
  mean_x.zeros();
  double cumsumw = 0;
  for(unsigned int i = 0; i < x.n_slices; i++) {
      double tmp = weights(i) + cumsumw;
      arma::mat diff = x.slice(i) - mean_x;
      mean_x += diff * weights(i) / tmp;
      for (unsigned int t = 0; t < x.n_cols; t++) {
        cov_x.slice(t) +=  weights(i) * diff.col(t) * (x.slice(i).col(t) - mean_x.col(t)).t();
      }
      cumsumw = tmp;
    }
  cov_x = cov_x / cumsumw * x.n_slices / (x.n_slices - 1);
}