
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
void conditional_dist_helper(arma::cube& V, arma::cube& C) {
  
  for (int t = V.n_slices - 1; t > 0; t--) {
    arma::mat tmp(V.n_cols, V.n_cols, arma::fill::zeros);
    if(any(V.slice(t - 1).diag() > 0.0)) {
      tmp = C.slice(t-1).t() * arma::pinv(V.slice(t - 1));
      V.slice(t) -= tmp * C.slice(t-1);
    }
    arma::uvec nonzero = arma::find(V.slice(t).diag() > 0);
    arma::mat L(V.n_cols, V.n_cols, arma::fill::zeros);
    if (nonzero.n_elem > 0) {
      L.submat(nonzero, nonzero) =
        arma::chol(V.slice(t).submat(nonzero, nonzero), "lower");
    }
    V.slice(t) = L;
    C.slice(t) = tmp;
  }
  arma::uvec nonzero = arma::find(V.slice(0).diag() > 0);
  arma::mat L(V.n_cols, V.n_cols, arma::fill::zeros);
  if (nonzero.n_elem > 0) {
    L.submat(nonzero, nonzero) =
      arma::chol(V.slice(0).submat(nonzero, nonzero), "lower");
  }
  V.slice(0) = L;
}
