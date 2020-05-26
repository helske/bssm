// back-tracking for filter smoother

#include "filter_smoother.h"

void filter_smoother(arma::cube& alpha, const arma::umat& indices) {
  
  arma::uvec b = arma::regspace<arma::uvec>(0, alpha.n_slices - 1);
  
  for (int t = alpha.n_cols - 2; t >= 0; t--) {
    arma::mat alphatmp = alpha.tube(arma::span::all, arma::span(t));
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      alpha.slice(i).col(t) = alphatmp.col(indices(b(i), t));
    }
    arma::uvec btmp = arma::vectorise(indices.col(t));
    b = btmp.rows(b);
  }

}
