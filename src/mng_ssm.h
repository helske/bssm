// multivariate nongaussian state space model with time-varying functions

#ifndef MNG_SSM_H
#define MNG_SSM_H

#include <sitmo.h>
#include "bssm.h"
#include "mgg_ssm.h"

// typedef for a pointer of linear function of ng-model equation returning Z, T, and R
typedef arma::mat (*lmat_fnPtr)(const unsigned int t, const arma::vec& theta, 
                   const arma::vec& known_params, const arma::mat& known_tv_params);
// typedef for a pointer returning a1
typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer returning P1
typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer of log-prior function
typedef double (*prior_fnPtr)(const arma::vec&);



class mng_ssm {
  
public:
  
  mng_ssm(const arma::mat& y, lmat_fnPtr Z_fn_, lmat_fnPtr T_fn_, 
    lmat_fnPtr R_fn_, a1_fnPtr a1_fn_, P1_fnPtr P1_fn_,
    const arma::vec& theta, prior_fnPtr log_prior_pdf_, const arma::vec& known_params, 
    const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
    const arma::uvec& time_varying, const arma::vec& phi, const arma::mat& u,
    const arma::uvec& distribution, const arma::uvec& phi_est, const arma::mat& initial_mode, 
    const bool local_approx = true, const unsigned int seed = 1, 
    const double zero_tol = 1e-8);
  
  arma::mat y;
  // functions of Z, T, R, a1 and P1
  lmat_fnPtr Z_fn;
  lmat_fnPtr T_fn;
  lmat_fnPtr R_fn;
  //initial value
  a1_fnPtr a1_fn;
  P1_fnPtr P1_fn;
  
  // Parameter vector used in _all_ linear functions
  arma::vec theta;
  //prior log-pdf
  prior_fnPtr log_prior_pdf;
  // vector of known parameters
  arma::vec known_params;
  // matrix of known (time-varying) parameters
  arma::mat known_tv_params;
  
  const unsigned int m;
  const unsigned int k;
  const unsigned int n;
  const unsigned int p;
  
  const unsigned int Ztv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  
  arma::vec phi;
  arma::mat u;
  const arma::uvec distribution;
  const arma::uvec phi_est;
  
  sitmo::prng_engine engine;
  const double zero_tol;
  
  const arma::mat initial_mode; // creating approx always starts from here
  arma::mat mode_estimate; // current estimate of mode
  unsigned int max_iter;
  double conv_tol;
  const bool local_approx;
  // -1 = no approx, 0 = theta doesn't match, 1 = proper local/global approx 
  int approx_state; 
  // store the current approx_loglik in order to avoid computing it again
  double approx_loglik; 
  // store the current scaling factors for PF/IS
  arma::vec scales;
  mgg_ssm approx_model;
  
  arma::vec log_likelihood(
      const unsigned int method, 
      const unsigned int nsim_states, 
      arma::cube& alpha, 
      arma::mat& weights, 
      arma::umat& indices);
  
  void update_model(const arma::vec& new_theta) {
    theta = new_theta;
    approx_state = 0;
  };
  // update the approximating Gaussian model
  void approximate();
  
  double compute_const_term() const;
  
  // given the mode_estimate, compute y and H of the approximating Gaussian model
  void laplace_iter(const arma::mat& signal, arma::mat& approx_y, 
    arma::cube& approx_H) const;
  
  Rcpp::List predict_interval(const arma::vec& probs, const arma::mat& thetasim,
    const arma::mat& alpha_last, const arma::cube& P_last,
    const arma::uvec& counts, const unsigned int predict_type);

  arma::cube predict_sample(const arma::mat& thetasim, const arma::mat& alpha,
    const arma::uvec& counts, const unsigned int predict_type,
    const unsigned int nsim);

  arma::cube sample_model(const arma::vec& a1_sim,
    const unsigned int predict_type, const unsigned int nsim);

    // bootstrap filter
  double bsf_filter(const unsigned int nsim, arma::cube& alpha,
    arma::mat& weights, arma::umat& indices);

  // psi-particle filter
  double psi_filter(const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const unsigned int t, const arma::cube& alphasim) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  void update_scales();
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alpha) const;
};


#endif
