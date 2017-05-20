// multivariate state space model with non-Gaussian or non-linear observation equation
// and linear Gaussian states

#ifndef MNG_H
#define MNG_H

#include "mnn_abc.h"
#include "bssm.h"

class mgg_ssm;

class mng_ssm: public mnn_abc {
  
public:
  
  // constructor from Rcpp::List
  mng_ssm(const Rcpp::List& model, 
    const unsigned int seed = 1, 
    const arma::uvec& Z_ind = arma::uvec(),
    const arma::uvec& T_ind = arma::uvec(), 
    const arma::uvec& R_ind = arma::uvec());
  
  //get and set theta
  void set_theta(const arma::vec& theta);
  arma::vec get_theta() const;
  
  // compute covariance matrices RR and regression part
  void compute_RR();
  void compute_xbeta() { xbeta = xreg * beta; }
  
  // compute y and H of the approximating Gaussian model
  void laplace_iter(const arma::mat& signal, arma::mat& approx_y, 
    arma::cube& approx_H) const;
  // find the approximating Gaussian model
  mgg_ssm approximate(arma::mat& signal, const unsigned int max_iter, 
    const double conv_tol) const;
  // psi-particle filter
  double psi_filter(const mgg_ssm& approx_model,
    const double approx_loglik, const arma::vec& scales,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  double log_likelihood(const unsigned int method, const unsigned int nsim_states);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec weights(const mgg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scaling_factors(const mgg_ssm& approx_model, const arma::vec& mode_estimate) const;
    
  arma::mat y;
  arma::cube Z;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::mat xreg;
  arma::vec beta;
  arma::mat C;
  arma::mat D;
  
  const unsigned int Ztv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Ctv;
  const unsigned int Dtv;
  
  const unsigned int n;
  const unsigned int m;
  const unsigned int k;
  const unsigned int p;
  
  arma::cube RR;
  arma::vec xbeta;
  
  std::mt19937 engine;
  const double zero_tol;
  
  double phi;
  arma::mat u;
  unsigned int distribution;
  bool phi_est;
  unsigned int max_iter;
  double conv_tol;
  
private:
  arma::uvec Z_ind;
  arma::uvec T_ind;
  arma::uvec R_ind;
  unsigned int seed;
};



#endif
