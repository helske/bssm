#include "conditional_dist.h"

// [[Rcpp::export]]
void conditional_cov(arma::cube& Vt, arma::cube& Ct) {
  
  unsigned int p = Vt.n_cols;
  for (int t = Vt.n_slices - 1; t > 0; t--) {
    arma::mat U(p, p);
    arma::mat V(1, 1); //not using this
    arma::vec s(p);
    arma::svd_econ(U, s, V, Vt.slice(t - 1), "left");
    arma::uvec nonzero = arma::find(s > (arma::datum::eps * p * s(0)));
    arma::mat tmp = Ct.slice(t - 1).t() * U.cols(nonzero) * 
      arma::diagmat(1.0 / s(nonzero)) * U.cols(nonzero).t();
    Vt.slice(t) -= tmp * Ct.slice(t - 1);
    Ct.slice(t) = tmp;
    
    arma::svd_econ(U, s, V, Vt.slice(t), "left");
    
    Vt.slice(t) = U * arma::diagmat(arma::sqrt(s));
    
  }
  arma::mat U(p, p);
  arma::mat V(1, 1); //not using this
  arma::vec s(p);
  arma::svd_econ(U, s, V, Vt.slice(0), "left");
  
  Vt.slice(0) = U * arma::diagmat(arma::sqrt(s));
}
