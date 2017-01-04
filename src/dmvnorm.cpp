#include <RcppArmadillo.h>
double dmvnorm(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr, bool logd) { 
  
  double out = -arma::datum::inf;
  
  unsigned int p = x.n_elem;
  
  arma::mat U(p, p);
  arma::mat V(p, p);
  arma::vec s(p);
  bool success = arma::svd_econ(U, s, V, sigma, "left");
  
  if (success) {
    arma::uvec nonzero = arma::find(s > (arma::datum::eps * p * s(0)));
    arma::vec tmp = U.cols(nonzero).t() * (x - mean);
    if (lwr) {
      s(nonzero) %= s(nonzero);
    } 
    out = -0.5 * arma::as_scalar(tmp.t() * arma::diagmat(1.0 / s(nonzero)) * tmp);
  }
  
  if (!logd) {
    out = exp(out);
  }
  
  return(out);
}
