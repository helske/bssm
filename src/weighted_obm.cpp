#include "bssm.h"
// [[Rcpp::export]]
double weighted_obm(const arma::vec& x, const arma::vec w) {
  
  unsigned int b;
  unsigned int n = x.n_elem;
  
  b = std::floor(std::sqrt(static_cast <double> (n)));
  
  unsigned int a = n - b + 1;
  
  arma::vec y(a);
  arma::vec wsum(a);
  for(unsigned int i = 0; i < a; i++) {
    wsum(i) = arma::accu(w.subvec(i, i + b - 1));
    y(i) = arma::accu(x.subvec(i, i + b - 1) % w.subvec(i, i + b - 1)) / wsum(i);
  }
  double wsum_all = arma::accu(w);
  double mu = arma::sum(x % w) / wsum_all;
  return std::sqrt(n * arma::accu(arma::square(wsum / arma::sum(wsum))) / wsum_all * 
    arma::accu(wsum % arma::square(y - mu)) / (a - 1.0));
}
