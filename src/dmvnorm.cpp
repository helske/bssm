#include "dmvnorm.h"

//[[Rcpp::export]]
double dmvnorm(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr, bool logd) { 
  
  double out = -std::numeric_limits<double>::infinity();
  
  if (x.n_elem == 1) {
    if (lwr) {
      out = -0.5 * (std::log(2.0 * M_PI) + std::log(sigma(0)) + std::pow((x(0) - mean(0)) / sigma(0), 2));
    } else {
      out = -std::log(sigma(0)) - 0.5 * (std::log(2.0 * M_PI) + std::pow(x(0) - mean(0), 2) / sigma(0));
    }
  }
  
  arma::uvec finite_x = arma::find_finite(x);
  unsigned int p = finite_x.n_elem;
  
  if (lwr) {
    // lazy, could we use previous cholesky?
    if (p < x.n_elem) {
      
      arma::mat sigma2 = sigma * sigma.t();
      arma::mat U(p, p);
      arma::mat V(p, p);
      arma::vec s(p);
      bool success = arma::svd_econ(U, s, V, sigma2(finite_x, finite_x), "left");
      
      if (success) {
        arma::uvec nonzero = arma::find(s > (std::numeric_limits<double>::epsilon() * p * s(0)));
        
        arma::vec tmp = U.cols(nonzero).t() * (x(finite_x) - mean((finite_x)));
        out = -0.5 * (nonzero.n_elem * std::log(2.0 * M_PI) + arma::accu(arma::log(s(nonzero))) + 
          arma::as_scalar(tmp.t() * arma::diagmat(1.0 / s(nonzero)) * tmp));
      }
    } else {
      arma::uvec nonzero = arma::find(sigma.diag() > (std::numeric_limits<double>::epsilon() * p * sigma.diag().max()));
      if (finite_x.n_elem < p) {
        nonzero = arma::intersect(finite_x, nonzero);
      }
      arma::mat S = inv(trimatl(sigma(nonzero, nonzero)));
      arma::vec tmp = S * (x.rows(nonzero) - mean.rows(nonzero));
      
      out = -0.5 * (nonzero.n_elem * std::log(2.0 * M_PI) + 
        2.0 * arma::accu(arma::log(arma::diagvec(sigma.submat(nonzero, nonzero)))) + 
        arma::as_scalar(tmp.t() * tmp));
    }
  } else {
    arma::mat U(p, p);
    arma::mat V(p, p);
    arma::vec s(p);
    bool success = arma::svd_econ(U, s, V, sigma(finite_x, finite_x), "left");
    
    if (success) {
      arma::uvec nonzero = arma::find(s > (std::numeric_limits<double>::epsilon() * p * s(0)));
      
      arma::vec tmp = U.cols(nonzero).t() * (x(finite_x) - mean((finite_x)));
      out = -0.5 * (nonzero.n_elem * std::log(2.0 * M_PI) + arma::accu(arma::log(s(nonzero))) + 
        arma::as_scalar(tmp.t() * arma::diagmat(1.0 / s(nonzero)) * tmp));
    }
  }
  
  if (!logd) {
    out = std::exp(out);
  }
  
  return(out);
}

//[[Rcpp::export]]
double precompute_dmvnorm(const arma::mat& sigma, arma::mat& Linv, const arma::uvec& nonzero) { 
  
  Linv = arma::inv(arma::trimatl(sigma(nonzero, nonzero)));
  double constant = -0.5 * nonzero.n_elem * std::log(2.0 * M_PI) + 
    arma::accu(arma::log(Linv.diag()));
  return constant;
}
//[[Rcpp::export]]
double fast_dmvnorm(const arma::vec& x, const arma::vec& mean, 
  const arma::mat& Linv, const arma::uvec& nonzero, const double constant) { 
  
  arma::vec tmp = Linv * (x.rows(nonzero) - mean.rows(nonzero));
  return constant - 0.5 * arma::accu(tmp % tmp);
}

