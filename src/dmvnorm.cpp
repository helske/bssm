//http://gallery.rcpp.org/articles/dmvnorm_arma/
//with small modifications
#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
double dmvnorm1(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr, bool logd) { 
  
  //review this
  arma::uvec nz = arma::find(arma::abs(arma::sum(sigma,1)) > 0);
  arma::mat L(nz.n_elem, nz.n_elem);
  if (!lwr) {
    L = arma::chol(sigma.submat(nz,nz));
  } else {
    L = sigma.rows(nz);
  }
  arma::mat rooti = arma::trans(arma::inv(arma::trimatu(L)));
  double rootisum = arma::sum(log(rooti.diag()));
  double c = -0.5 * nz.n_elem * std::log(2.0 * M_PI);
  arma::vec z = rooti * (x.elem(nz) - mean.elem(nz));    
  double out = c - 0.5 * arma::sum(z%z) + rootisum;     
  
  if (!logd) {
    out = exp(out);
  }
  return(out);
}

// [[Rcpp::export]]
arma::vec dmvnorm2(const arma::mat& x, const arma::mat& mean,  
  arma::mat sigma, bool lwr, bool logd, const arma::mat& A) { 
  
  //review this
  arma::vec out(x.n_cols);
  arma::uvec nz = arma::find(arma::abs(arma::sum(sigma,1)) > 0);
  arma::mat L(nz.n_elem, nz.n_elem);
  if (!lwr) {
    L = arma::chol(sigma.submat(nz,nz));
  } else {
    L = sigma.rows(nz);
  }
  arma::mat rooti = arma::trans(arma::inv(arma::trimatu(L)));
  double rootisum = arma::sum(log(rooti.diag()));
  double c = -0.5 * nz.n_elem * std::log(2.0 * M_PI);
  
  for (unsigned int i=0; i < x.n_cols; i++) {
    arma::vec m = A * mean.col(i);
    arma::vec y = x.col(i);
    arma::vec z = rooti * (y.elem(nz) - m.elem(nz));    
    out(i) = c - 0.5 * arma::sum(z%z) + rootisum;     
  }  
  
  if (!logd) {
    out = exp(out);
  }
  return(out);
}

