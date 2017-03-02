// univariate state space model with non-Gaussian or non-linear observation equation
// and linear Gaussian states

#ifndef UNG_SSM_H
#define UNG_SSM_H

#include "unn_abc.h"
#include <RcppArmadillo.h>

class ugg_ssm;

class ung_ssm: public unn_abc {
  
public:
  
  // constructor from Rcpp::List
  ung_ssm(const Rcpp::List& model, 
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
  void laplace_iter(const arma::vec& mode_estimate, arma::vec& approx_y, 
    arma::vec& approx_H) const;
  // find the approximating Gaussian model
  ugg_ssm approximate(arma::vec& mode_estimate, const unsigned int max_iter, 
    const double conv_tol) const;
  
  // update aproximating Gaussian model
  void approximate(ugg_ssm& approx_model, arma::vec& mode_estimate, 
    const unsigned int max_iter, const double conv_tol) const;
  // psi-particle filter
  double psi_filter(const ugg_ssm& approx_model,
    const double approx_loglik, const arma::vec& scales,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const ugg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scaling_factors(const ugg_ssm& approx_model, const arma::vec& mode_estimate) const;
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alphasim) const;
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, arma::cube& alphasim, 
      arma::mat& weights, arma::umat& indices);
  
  arma::cube predict_sample(const arma::mat& theta,
    const arma::mat& alpha, const arma::uvec& counts, const bool predict_obs);
  
  arma::mat sample_model(const bool simulate_obs);
  
  arma::vec y;
  arma::mat Z;
  arma::cube T;
  arma::cube R;
  arma::cube Q;
  arma::vec a1;
  arma::mat P1;
  arma::mat xreg;
  arma::vec beta;
  arma::vec D;
  arma::mat C;
  
  const unsigned int Ztv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Dtv;
  const unsigned int Ctv;
  
  const unsigned int n;
  const unsigned int m;
  const unsigned int k;
  
  arma::cube RR;
  arma::vec xbeta;
  
  std::mt19937 engine;
  const double zero_tol;
  
  double phi;
  arma::vec u;
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
