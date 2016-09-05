
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
void backtrack_pf(arma::cube& alpha, arma::umat& ind) {
  // arma::cube alphatmp2(alpha.n_rows, alpha.n_cols, alpha.n_slices);
  // for (unsigned int t = 0; t < alpha.n_cols - 1; t++) {
  //   for (unsigned int i = 0; i < alpha.n_slices; i++) {
  //     alphatmp2(arma::span::all, arma::span(0, t), arma::span(i)) = 
  //       alpha(arma::span::all, arma::span(0, t), arma::span(ind(i,t)));
  //   }
  //   alpha(arma::span::all, arma::span(0, t), arma::span::all) = 
  //     alphatmp2(arma::span::all, arma::span(0, t), arma::span::all);
  // } 
  
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
