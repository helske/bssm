// univariate state space model with non-Gaussian or non-linear observation equation
// and linear Gaussian states

#ifndef SSM_UNG_H
#define SSM_UNG_H

#include "bssm.h"
#include <sitmo.h>

#include "model_ssm_ulg.h"

class ssm_ung {
  
public:
  
  // constructor from Rcpp::List
  ssm_ung(const Rcpp::List model, 
    const unsigned int seed = 1,
    const double zero_tol = 1e-8);
  
  arma::vec y;
  arma::mat Z;
  arma::cube T;
  arma::cube R;
  arma::cube Q;
  arma::vec a1;
  arma::mat P1;
  arma::vec D;
  arma::mat C;
  arma::mat xreg;
  arma::vec beta;
  
  const unsigned int n; // number of time points
  const unsigned int m; // number of states
  const unsigned int k; // number of etas
  static const unsigned int p = 1; // number of series
  
  // is the matrix/vector time-varying?
  const unsigned int Ztv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Dtv;
  const unsigned int Ctv;
  
  arma::vec theta;  
  
  double phi;
  arma::vec u;
  const unsigned int distribution;
  unsigned int max_iter;
  double conv_tol;
  const bool local_approx;
  const arma::mat initial_mode; // creating approx always starts from here
  arma::mat mode_estimate; // current estimate of mode
  
  // -1 = no approx, 0 = theta doesn't match, 
  // 1 = proper local/global approx, 2 = approx_loglik updated
  int approx_state; 
  // store the current approx_loglik in order to avoid computing it again
  double approx_loglik; 
  // store the current scaling factors for PF/IS
  arma::vec scales;
  
  // random number engine
  sitmo::prng_engine engine;
  // zero-tolerance
  const double zero_tol;
  
  arma::cube RR;
  arma::vec xbeta;
  
  // R functions
  const Rcpp::Function update_fn;
  const Rcpp::Function prior_fn;
  
  ssm_ulg approx_model;
  
  virtual void update_model(const arma::vec& new_theta);
  virtual double log_prior_pdf(const arma::vec& x) const;
  void compute_RR(){
    for (unsigned int t = 0; t < R.n_slices; t++) {
      RR.slice(t) = R.slice(t) * R.slice(t).t();
    }
  }
  void compute_xbeta() { xbeta = xreg * beta; }
  
  arma::vec log_likelihood(
      const unsigned int method, 
      const unsigned int nsim, 
      arma::cube& alpha, 
      arma::mat& weights, 
      arma::umat& indices);
  
  // update approximating Gaussian model
  void approximate();
  void approximate_for_is(const arma::mat& mode_estimate_);
  // given the mode_estimate, compute y and H of the approximating Gaussian model
  void laplace_iter(const arma::vec& signal);

  // psi-particle filter
  double psi_filter(const unsigned int nsim, arma::cube& alpha, 
    arma::mat& weights, arma::umat& indices);
  
  // compute log-weights over all time points (see below)
  arma::vec importance_weights(const arma::cube& alpha) const;
    
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const unsigned int t, const arma::cube& alphasim) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  void update_scales();
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alphasim) const;
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, arma::cube& alphasim, 
      arma::mat& weights, arma::umat& indices);
  
  double compute_const_term(); 
  
  arma::cube predict_sample(const arma::mat& theta_posterior, const arma::mat& alpha, 
    const unsigned int predict_type);
  
  arma::mat sample_model(const unsigned int predict_type);

};



#endif
