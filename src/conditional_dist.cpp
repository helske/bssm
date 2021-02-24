#include "conditional_dist.h"

// [[Rcpp::export]]
void conditional_cov(arma::cube& Vt, arma::cube& Ct, const bool use_svd) {
  
  unsigned int p = Vt.n_cols;
  
  if (use_svd) {
    for (int t = Vt.n_slices - 1; t > 0; t--) {
      
      arma::mat U(p, p);
      arma::mat V(p, p); //not using this
      arma::vec s(p);
      arma::svd_econ(U, s, V, Vt.slice(t - 1), "left");
      arma::uvec nonzero = arma::find(s > (std::numeric_limits<double>::epsilon() * p * s(0)));
      arma::mat tmp = Ct.slice(t - 1).t() * U.cols(nonzero) * 
        arma::diagmat(1.0 / s(nonzero)) * U.cols(nonzero).t();
      Vt.slice(t) -= tmp * Ct.slice(t - 1);
      Ct.slice(t) = tmp;
      
      arma::svd_econ(U, s, V, Vt.slice(t), "left");
      
      Vt.slice(t) = U * arma::diagmat(arma::sqrt(s));
    }
    
    arma::mat U(p, p);
    arma::mat V(p, p); //not using this
    arma::vec s(p);
    arma::svd_econ(U, s, V, Vt.slice(0), "left");
    
    Vt.slice(0) = U * arma::diagmat(arma::sqrt(s));
    
  } else {
    for (int t = Vt.n_slices - 1; t > 0; t--) {
      // Vt can be singular if the states contain deterministic components
      arma::vec diagV = Vt.slice(t - 1).diag();
      arma::uvec nonzero = 
        arma::find(diagV > std::numeric_limits<double>::epsilon());
      
      arma::mat cholVsub = arma::chol(Vt.slice(t - 1).submat(nonzero, nonzero), "lower");
      
      // X = inv(cholV) * C  
      arma::mat tmp = arma::solve(arma::trimatl(cholVsub), Ct.slice(t - 1).submat(nonzero, nonzero));
      // Vt = Vt - C'*inv(Vt-1)*C
      Vt.slice(t).submat(nonzero, nonzero) -= tmp.t() * tmp;
      Ct.slice(t).submat(nonzero, nonzero) = arma::solve(arma::trimatl(cholVsub), tmp).t();
      
      arma::vec diagV2 = Vt.slice(t).diag();
      arma::uvec nonzero2 = 
        arma::find(diagV2 > std::numeric_limits<double>::epsilon());
      
      arma::mat cholVsub2 = arma::chol(Vt.slice(t).submat(nonzero2, nonzero2), "lower");
      
      Vt.slice(t).zeros();
      Vt.slice(t).submat(nonzero2, nonzero2) = cholVsub2;
    }
    arma::vec diagV = Vt.slice(0).diag();
    arma::uvec nonzero = 
      arma::find(diagV > std::numeric_limits<double>::epsilon());
    
    arma::mat cholV(p, p, arma::fill::zeros);
    arma::mat cholVsub = arma::chol(Vt.slice(0).submat(nonzero, nonzero), "lower");
    cholV.submat(nonzero, nonzero) = cholVsub;
    Vt.slice(0) = cholV;
  }
  
}
