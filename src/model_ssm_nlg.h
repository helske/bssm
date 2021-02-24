// arbitrary nonlinear gaussian state space model with time-varying functions
#ifndef SSM_NLG_H
#define SSM_NLG_H

#include <sitmo.h>
#include "bssm.h"
#include "model_ssm_mlg.h"

// typedef for a pointer of nonlinear function of model equation returning vec (T, Z)
typedef arma::vec (*nvec_fnPtr)(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);
// typedef for a pointer of nonlinear function of model equation returning mat (Tg, Zg, H, R)
typedef arma::mat (*nmat_fnPtr)(const unsigned int t, const arma::vec& alpha, const arma::vec& theta, 
  const arma::vec& known_params, const arma::mat& known_tv_params);

// typedef for a pointer returning a1
typedef arma::vec (*a1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer returning P1
typedef arma::mat (*P1_fnPtr)(const arma::vec& theta, const arma::vec& known_params);
// typedef for a pointer of log-prior function
typedef double (*prior_fnPtr)(const arma::vec& theta);

class ssm_nlg {
  
public:
  
  ssm_nlg(
    const arma::mat& y, 
    nvec_fnPtr Z_fn_, 
    nmat_fnPtr H_fn_, 
    nvec_fnPtr T_fn_, 
    nmat_fnPtr R_fn_, 
    nmat_fnPtr Z_gn_, 
    nmat_fnPtr T_gn_, 
    a1_fnPtr a1_fn_, 
    P1_fnPtr P1_fn_,
    const arma::vec& theta, 
    prior_fnPtr log_prior_pdf_, 
    const arma::vec& known_params, 
    const arma::mat& known_tv_params, 
    const unsigned int m, 
    const unsigned int k,
    const arma::uvec& time_varying,
    const unsigned int seed = 1,
    const unsigned int iekf_iter = 0,
    const unsigned int max_iter = 100,
    const double conv_tol = 1e-8);
  
  arma::mat y;
  // nonlinear functions of 
  // y_t = Z(alpha_t, theta,t) + H(theta,t)*eps_t, 
  // alpha_t+1 = T(alpha_t, theta,t) + R(theta, t)*eta_t
  
  nvec_fnPtr Z_fn;
  nmat_fnPtr H_fn;
  nvec_fnPtr T_fn;
  nmat_fnPtr R_fn;
  //and the derivatives
  nmat_fnPtr Z_gn;
  nmat_fnPtr T_gn;
  //initial value
  a1_fnPtr a1_fn;
  P1_fnPtr P1_fn;
  
  // Parameter vector used in _all_ nonlinear functions
  arma::vec theta;
  //prior log-pdf
  prior_fnPtr log_prior_pdf_;
  // vector of known parameters
  arma::vec known_params;
  // matrix of known (time-varying) parameters
  arma::mat known_tv_params;
  
  const unsigned int m;
  const unsigned int k;
  const unsigned int n;
  const unsigned int p;
  
  const unsigned int Zgtv;
  const unsigned int Htv;
  const unsigned int Tgtv;
  const unsigned int Rtv;
  
  unsigned int seed;
  sitmo::prng_engine engine;
  const double zero_tol;
  
  unsigned int iekf_iter;
  unsigned int max_iter;
  double conv_tol;
  
  arma::mat mode_estimate; // current estimate of the mode
  // -1 = no approx, 0 = theta doesn't match, 1 = proper approx 
  int approx_state; 
  // store the current approx_loglik in order to avoid computing it again
  double approx_loglik; 
  // store the current scaling factors for psi-APF
  arma::vec scales;
  ssm_mlg approx_model;
  
  void update_model(const arma::vec& new_theta);
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const;
  // update the approximating Gaussian model
  void approximate();
  void approximate_for_is(const arma::mat& mode_estimate);
  void approximate_by_ekf();
  
  arma::vec log_likelihood(
      const unsigned int method, 
      const unsigned int nsim, 
      arma::cube& alpha, 
      arma::mat& weights, 
      arma::umat& indices);
  
  double ekf(arma::mat& at, arma::mat& att, arma::cube& Pt, 
    arma::cube& Ptt) const;
  
  double ekf_loglik() const;
  
  double ekf_smoother(arma::mat& att, arma::cube& Ptt) const;
  double ekf_fast_smoother(arma::mat& at) const;
  
  double ukf(arma::mat& at, arma::mat& att, arma::cube& Pt, arma::cube& Ptt, 
    const double alpha = 1.0, const double beta = 0.0, const double kappa = 2.0) const;
  
  // bootstrap filter  
  double bsf_filter(const unsigned int nsim, arma::cube& alpha, 
    arma::mat& weights, arma::umat& indices);
  
  // psi-particle filter
  double psi_filter(const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // extended Kalman particle filter
  double ekf_filter(const unsigned int nsim, arma::cube& alpha,
    arma::mat& weights, arma::umat& indices);
  
  void update_scales();
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const unsigned int t, const arma::cube& alpha, const arma::mat& alpha_prev) const;
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alpha) const;
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  double log_obs_density(const unsigned int t, const arma::vec& alpha) const;
  
  void ekf_update_step(const unsigned int t, const arma::vec y, 
    const arma::vec& at, const arma::mat& Pt, arma::vec& att, arma::mat& Ptt) const;
  
  double log_signal_pdf(const arma::mat& alpha) const;
  
  arma::cube predict_sample(const arma::mat& theta_posterior, 
    const arma::mat& alpha, const unsigned int predict_type);
  
  arma::mat sample_model(const arma::vec& a1_sim, 
    const unsigned int predict_type);
  
  arma::cube predict_past(const arma::mat& theta_posterior,
    const arma::cube& alpha, const unsigned int predict_type);
};


#endif
