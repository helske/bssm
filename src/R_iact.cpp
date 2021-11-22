#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double IACT(const arma::vec x) {
  
  unsigned int n = x.n_elem;
  double C = std::max(5.0, std::log10(n));
  double tau = 1.0;
 
  for (unsigned int k = 1; k < n; k++) {
    tau += 2.0 * arma::dot(x.subvec(0, n - k - 1), x.subvec(k, n - 1)) / (n - k);
    if (k > C * tau) break;
  } 
  return std::max(0.0, tau);
}
