// arbitrary nonlinear gaussian state space model with time-invariant functions

#ifndef NLG_SSM_H
#define NLG_SSM_H

#include "bssm.h"
#include "mgg_ssm.h"
#include "function_pointers.h"


class nlg_ssm {
  
public:
  
  nlg_ssm(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_, SEXP T_fn_, SEXP R_fn_, 
    SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
    const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params, 
    const arma::mat& known_tv_params, const unsigned int m, const unsigned int k,
    const arma::uvec& time_varying, const unsigned int seed);
  
  // find the approximating Gaussian model
  mgg_ssm approximate(arma::mat& mode_estimate, 
    const unsigned int max_iter, const double conv_tol, 
    const unsigned int iekf_iter) const;
  // update the approximating Gaussian model
  arma::mat approximate(mgg_ssm& approx_model, const unsigned int max_iter, 
    const double conv_tol) const;
  
  
  arma::cube predict_sample(const arma::mat& thetasim, const arma::mat& alpha, 
    const arma::uvec& counts, const unsigned int predict_type);
  arma::mat sample_model(const arma::vec& a1_sim, const unsigned int predict_type);
  
  double ekf(arma::mat& at, arma::mat& att, arma::cube& Pt, 
    arma::cube& Ptt, const unsigned int iekf_iter) const;
  
  double ekf_loglik(const unsigned int iekf_iter) const;
  
  double ekf_smoother(arma::mat& att, arma::cube& Ptt, const unsigned int iekf_iter) const;
  double ekf_fast_smoother(arma::mat& at, const unsigned int iekf_iter) const;
  
  double ukf(arma::mat& at, arma::mat& att, arma::cube& Pt, arma::cube& Ptt, 
    const double alpha = 1.0, const double beta = 0.0, const double kappa = 2.0) const;
  
    // bootstrap filter  
  double bsf_filter(const unsigned int nsim, arma::cube& alphasim, 
    arma::mat& weights, arma::umat& indices);
  
  // auxiliary particle filter  
  double aux_filter(const unsigned int nsim, arma::cube& alphasim, 
    arma::mat& weights, arma::umat& indices);
  
  // psi-particle filter
  double psi_filter(const mgg_ssm& approx_model, const double approx_loglik,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // extended Kalman particle filter
  double ekf_filter(const unsigned int nsim, arma::cube& alpha,
    arma::mat& weights, arma::umat& indices);
  
  // defensive psi-particle filter
  double df_psi_filter(const mgg_ssm& approx_model, const double approx_loglik,
    const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const mgg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim, const arma::mat& alpha_prev) const;
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights_df(const mgg_ssm& approx_model, 
    const unsigned int t, const arma::cube& alphasim, const arma::mat& alpha_prev) const;
  
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scaling_factors(const mgg_ssm& approx_model, const arma::mat& mode_estimate) const;
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alphasim) const;
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  double log_obs_density(const unsigned int t, const arma::vec& alphasim) const;
  
  void ekf_update_step(const unsigned int t, const arma::vec y, 
    const arma::vec& at, const arma::mat& Pt, arma::vec& att, arma::mat& Ptt) const;
    
  double log_signal_pdf(const arma::mat& alpha) const;
  
  arma::mat y;
  // nonlinear functions of 
  // y_t = Z(alpha_t, theta,t) + H(theta,t)*eps_t, 
  // alpha_t+1 = T(alpha_t, theta,t) + R(theta, t)*eta_t
  
  vec_fn Z_fn;
  mat_varfn H_fn;
  vec_fn T_fn;
  mat_varfn R_fn;
  //and the derivatives
  mat_fn Z_gn;
  mat_fn T_gn;
  //initial value
  vec_initfn a1_fn;
  mat_initfn P1_fn;
  
  // Parameter vector used in _all_ nonlinear functions
  arma::vec theta;
  //prior log-pdf
  double_fn log_prior_pdf;
  // vector of known parameters
  arma::vec known_params;
  // matrix of known (time-varying) parameters
  arma::mat known_tv_params;
  
  const unsigned int m;
  const unsigned int k;
  const unsigned int n;
  const unsigned int p;
  
  const unsigned int Zgtv;
  const unsigned int Tgtv;
  const unsigned int Htv;
  const unsigned int Rtv;
  
  unsigned int seed;
  std::mt19937 engine;
  const double zero_tol;
  
};


#endif
