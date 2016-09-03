
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
void backtrack_pf(arma::cube& alpha, arma::umat& ind) {
  for (int t = alpha.n_cols - 2; t >= 0; t--) {
    arma::mat alphatmp = alpha.tube(arma::span::all, arma::span(t));
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      alpha.slice(i).col(t) = alphatmp.col(ind(i, t));
    }
  }
}
