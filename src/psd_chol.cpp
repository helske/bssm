#include "psd_chol.h"
// Compute the Cholesky factor of positive semidefinite matrix where the 
// semidefiniteness is due to zeros on diagonal
// [[Rcpp::export]]
arma::mat psd_chol(const arma::mat& x) {
  
  arma::uvec nonzero = 
    arma::find(x.diag() > std::max(std::numeric_limits<double>::epsilon(), 
      std::numeric_limits<double>::epsilon() * x.n_cols * x.diag().max()));
  unsigned int k = nonzero.n_elem;
  
  arma::mat cholx(x.n_cols,x.n_cols, arma::fill::zeros);
  if (k > 0) {
    cholx.submat(nonzero, nonzero) = arma::chol(x.submat(nonzero, nonzero), "lower");
  }
  
  return cholx;
}
