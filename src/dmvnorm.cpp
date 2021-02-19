#include "dmvnorm.h"

//[[Rcpp::export]]
double dmvnorm(const arma::vec& x, const arma::vec& mean,  
  const arma::mat& sigma, bool lwr, bool logd) { 
  
  double out = -std::numeric_limits<double>::infinity();
  
  if (x.n_elem == 1) {
    if (lwr) {
      out = -0.5 * (std::log(2.0 * M_PI) + 2.0 * std::log(sigma(0)) + std::pow((x(0) - mean(0)) / sigma(0), 2));
    } else {
      out = - 0.5 * (std::log(2.0 * M_PI) + std::log(sigma(0)) + std::pow(x(0) - mean(0), 2) / sigma(0));
    }
  } else {
    
    // ignore rows/columns corresponding to missing values
    arma::uvec finite_x = arma::find_finite(x);
    unsigned int p = finite_x.n_elem;
    arma::mat Sx(p, p);
    arma::vec z = x(finite_x) - mean(finite_x);
    if (lwr) {
      arma::mat sigma2 = sigma * sigma.t();
      Sx = sigma2(finite_x, finite_x);
    } else {
      Sx = sigma(finite_x, finite_x);
    }
    arma::uvec nonzero = arma::find(Sx.diag() > std::numeric_limits<double>::epsilon());
    
    arma::mat S = Sx(nonzero, nonzero);
    
    arma::mat rooti = arma::inv(trimatl(arma::chol(S, "lower")));
    arma::vec z2 = rooti * z(nonzero);
    out = arma::sum(log(rooti.diag())) - 0.5 * S.n_rows * std::log(2.0 * M_PI) - 0.5 * arma::dot(z2, z2);     
    
    
    // if (lwr) {
    //   arma::mat sigma2 = sigma * sigma.t();
    //   // lazy, could we use previous cholesky?
    //   if (p < x.n_elem) {
    // 
    //     arma::mat U(p, p);
    //     arma::mat V(p, p);
    //     arma::vec s(p);
    //     bool success = arma::svd_econ(U, s, V, sigma2(finite_x, finite_x), "left");
    // 
    //     if (success) {
    //       arma::uvec nonzero = arma::find(s > (std::numeric_limits<double>::epsilon() * p * s(0)));
    // 
    //       arma::vec tmp = U.cols(nonzero).t() * (x(finite_x) - mean((finite_x)));
    //       out = -0.5 * (nonzero.n_elem * std::log(2.0 * M_PI) + arma::accu(arma::log(s(nonzero))) +
    //         arma::as_scalar(tmp.t() * arma::diagmat(1.0 / s(nonzero)) * tmp));
    //     }
    //   } else {
    // 
    //     arma::uvec nonzero = arma::find(sigma2.diag() > (std::numeric_limits<double>::epsilon() * p * sigma2.diag().max()));
    //     if (finite_x.n_elem < p) {
    //       nonzero = arma::intersect(finite_x, nonzero);
    //     }
    // 
    //     // before 5.2.2021
    //     //arma::mat S = inv(trimatl(sigma(nonzero, nonzero)));
    //     //arma::vec tmp = S * (x.rows(nonzero) - mean.rows(nonzero));
    //     // note sigma2 !
    //     arma::mat S = arma::chol(sigma2(nonzero, nonzero), "lower");
    //     arma::vec tmp = arma::inv(trimatl(S)) * (x.rows(nonzero) - mean.rows(nonzero));
    // 
    //     out = -0.5 * (nonzero.n_elem * std::log(2.0 * M_PI) +
    //       2.0 * arma::accu(arma::log(arma::diagvec(S))) +
    //       arma::accu(tmp % tmp));
    //   }
    // } else {
    //   arma::mat U(p, p);
    //   arma::mat V(p, p);
    //   arma::vec s(p);
    //   bool success = arma::svd(U, s, V, sigma(finite_x, finite_x));//, "left");
    // 
    //   if (success) {
    //     arma::uvec nonzero = arma::find(s > (std::numeric_limits<double>::epsilon() * p * s(0)));
    // 
    //     arma::vec tmp = U.cols(nonzero).t() * (x(finite_x) - mean((finite_x)));
    //     out = -0.5 * (nonzero.n_elem * std::log(2.0 * M_PI) + arma::accu(arma::log(s(nonzero))) +
    //       arma::as_scalar(tmp.t() * arma::diagmat(1.0 / s(nonzero)) * tmp));
    //   }
    // }
  }
  if (!logd) {
    out = std::exp(out);
  }

  return(out);
}

//[[Rcpp::export]]
double precompute_dmvnorm(const arma::mat& sigma, arma::mat& Linv, const arma::uvec& nonzero) { 
  
  // before 5.2.2021
  // Linv = arma::inv(arma::trimatl(sigma(nonzero, nonzero)));
  // Can't assume sigma is triangular even though that is the purpose of defining H vs HH..
  
  Linv = arma::inv(trimatl(arma::chol(sigma(nonzero, nonzero), "lower")));
  
  double constant = -0.5 * nonzero.n_elem * std::log(2.0 * M_PI) +
    arma::accu(arma::log(Linv.diag()));
  return constant;
}
//[[Rcpp::export]] 
double fast_dmvnorm(const arma::vec& x, const arma::vec& mean, 
  const arma::mat& Linv, const arma::uvec& nonzero, const double constant) { 
  
  // note no missing observations allowed
  arma::vec tmp = Linv * (x.rows(nonzero) - mean.rows(nonzero));
  return constant - 0.5 * arma::accu(tmp % tmp);
}

