//back-tracking for filter smoother

#include <RcppArmadillo.h>

void backtrack_pf(arma::cube& alpha, const arma::umat& ind) {
  
  arma::uvec b(alpha.n_slices);
  for(unsigned int i = 0; i < b.n_elem; i++) {
    b(i) = i;
  }
  for (int t = alpha.n_cols - 2; t >= 0; t--) {
    arma::mat alphatmp = alpha.tube(arma::span::all, arma::span(t));
    for (unsigned int i = 0; i < alpha.n_slices; i++) {
      alpha.slice(i).col(t) = alphatmp.col(ind(b(i), t));
    }
    arma::uvec btmp = arma::vectorise(ind.col(t));
    b = btmp.rows(b);
  }

}
