// stratified sampling of indices from 0 to length(p)-1
// modified to armadillo compatible from C code by Matti Vihola
#include "sample.h"

// p is the target distribution
// r are random number from U(0,1)
// N is the number of samples
//[[Rcpp::export()]]
arma::uvec stratified_sample(arma::vec& p, const arma::vec& r, const unsigned int N) {

  arma::uvec xp(N);
  p = arma::cumsum(p);
  p(p.n_elem - 1) = 1;
  
  unsigned int j = 0;
  double alpha = 1.0/N;
  for(unsigned int k = 0; k < p.n_elem && j < N; k++) {
    while (j < N && (r(j) + j) * alpha <= p(k)) {
      xp(j) = k;
      j++;
    }
  }
  while (j < N) {
    xp(j) = p.n_elem - 1;
    j++;
  }
  return xp;
}
