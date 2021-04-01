// general SV model with stochastic mean and variance process

#ifndef SSM_GSV_H
#define SSM_GSV_H

#include <sitmo.h>
#include "bssm.h"
#include "model_ssm_mlg.h"

extern Rcpp::Function default_update_fn;

class ssm_gsv {
  
public:
  
  ssm_gsv(const Rcpp::List model, 
    const unsigned int seed = 1,
    const double zero_tol = 1e-12);
  
  arma::vec y;
  arma::mat Z_mu;
  arma::cube T_mu;
  arma::cube R_mu;
  arma::vec a1_mu;
  arma::mat P1_mu;
  arma::vec D_mu;
  arma::mat C_mu;
  
  arma::mat Z_sv;
  arma::cube T_sv;
  arma::cube R_sv;
  arma::vec a1_sv;
  arma::mat P1_sv;
  arma::vec D_sv;
  arma::mat C_sv;
  
  
  const unsigned int n; // number of time points
  const unsigned int m_mu; // number of states
  const unsigned int k_mu; // number of etas
  const unsigned int m_sv; // number of states
  const unsigned int k_sv; // number of etas
  const unsigned int m; // number of states
  const unsigned int k; // number of etas
  static const unsigned int p = 1; // number of series
  
  // is the matrix/vector time-varying?
  const unsigned int Ztv_mu;
  const unsigned int Ttv_mu;
  const unsigned int Rtv_mu;
  const unsigned int Dtv_mu;
  const unsigned int Ctv_mu;
  
  const unsigned int Ztv_sv;
  const unsigned int Ttv_sv;
  const unsigned int Rtv_sv;
  const unsigned int Dtv_sv;
  const unsigned int Ctv_sv;
  
  arma::vec theta; 
  
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
  
  sitmo::prng_engine engine;
  const double zero_tol;
  arma::cube RR_mu;
  arma::cube RR_sv;
  
  ssm_mlg approx_model;
  
  void compute_RR(){
    for (unsigned int t = 0; t < R_mu.n_slices; t++) {
      RR_mu.slice(t) = R_mu.slice(t) * R_mu.slice(t).t();
    }
    for (unsigned int t = 0; t < R_sv.n_slices; t++) {
      RR_sv.slice(t) = R_sv.slice(t) * R_sv.slice(t).t();
    }
  }
  
  arma::vec log_likelihood(
      const unsigned int method, 
      const unsigned int nsim, 
      arma::cube& alpha, 
      arma::mat& weights, 
      arma::umat& indices);
  
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const;
  
  // update the approximating Gaussian model
  void joint_model();
  unsigned int approximate();
  void approximate_for_is(const arma::mat& mode_estimate_);
  
  double compute_const_term() const;
  
  // given the mode_estimate, compute y and H of the approximating Gaussian model
  void laplace_iter(const arma::mat& signal);
  
  // bootstrap filter
  double bsf_filter(const unsigned int nsim, arma::cube& alpha,
    arma::mat& weights, arma::umat& indices);
  
  // psi-particle filter
  double psi_filter(const unsigned int nsim, arma::cube& alpha, arma::mat& weights,
    arma::umat& indices);
  
  // compute logarithms of _unnormalized_ importance weights g(y_t | alpha_t) / ~g(~y_t | alpha_t)
  arma::vec log_weights(const unsigned int t, const arma::cube& alphasim) const;
  arma::vec importance_weights(const arma::cube& alpha) const;
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  void update_scales();
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alpha) const;
  
  // arma::cube predict_sample(const arma::mat& theta_posterior, const arma::mat& alpha,
  //   const unsigned int predict_type, const Rcpp::Function update_fn = default_update_fn);
  // 
  // arma::mat sample_model(const unsigned int predict_type);
  // 
  // arma::cube predict_past(const arma::mat& theta_posterior,
  //   const arma::cube& alpha, const unsigned int predict_type, 
  //   const Rcpp::Function update_fn = default_update_fn);
};


#endif
