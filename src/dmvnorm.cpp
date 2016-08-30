//http://gallery.rcpp.org/articles/dmvnorm_arma/
//with small modifications
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
arma::vec dmvnorm1(const arma::mat& x, const arma::vec& mean,  
  arma::mat sigma, bool lwr = false, bool logd = false) { 
  
  arma::vec out(x.n_cols);
  if (!lwr) {
    sigma = arma::chol(sigma);
  }
  arma::mat rooti = arma::trans(arma::pinv(arma::trimatu(sigma)));
  double rootisum = arma::sum(log(rooti.diag()));
  double c = -0.5 * x.n_rows * std::log(2.0 * M_PI);
  
  for (int i=0; i < x.n_cols; i++) {
    arma::vec z = rooti * (x.col(i) - mean) ;    
    out(i) = c - 0.5 * arma::sum(z%z) + rootisum;     
  }  
  
  if (!logd) {
    out = exp(out);
  }
  return(out);
}

// [[Rcpp::export]]
arma::vec dmvnorm2(const arma::mat& x, const arma::mat& mean,  
  arma::mat sigma, bool lwr, bool logd, const arma::mat& A) { 
  
  arma::vec out(x.n_cols);
  if (!lwr) {
    sigma = arma::chol(sigma);
  }
  arma::mat rooti = arma::trans(arma::pinv(arma::trimatu(sigma)));
  double rootisum = arma::sum(log(rooti.diag()));
  double c = -0.5 * x.n_rows * std::log(2.0 * M_PI);
  
  for (int i=0; i < x.n_cols; i++) {
    arma::vec z = rooti * (x.col(i) - A * mean.col(i)) ;    
    out(i) = c - 0.5 * arma::sum(z%z) + rootisum;     
  }  
  
  if (!logd) {
    out = exp(out);
  }
  return(out);
}

