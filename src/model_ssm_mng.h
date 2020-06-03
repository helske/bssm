// multivariate  state space model with non-Gaussian or non-linear observation equation
// and linear Gaussian states

#ifndef SSM_MNG_H
#define SSM_MNG_H

#include <sitmo.h>
#include "bssm.h"
#include "model_ssm_mlg.h"

class ssm_mng {
  
public:
  
  ssm_mng(const Rcpp::List model, 
    const unsigned int seed = 1,
    const double zero_tol = 1e-8);
  
  arma::mat y;
  arma::cube Z;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::mat D;
  arma::mat C;
  
  const unsigned int n; // number of time points
  const unsigned int m; // number of states
  const unsigned int k; // number of etas
  const unsigned int p; // number of series
  
  // is the matrix/vector time-varying?
  const unsigned int Ztv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Dtv;
  const unsigned int Ctv;
  
  arma::vec theta; 
  
  arma::vec phi;
  arma::mat u;
  const arma::uvec distribution;
  const arma::uvec phi_est;
  
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
  arma::cube RR;
  
  // R functions
  const Rcpp::Function update_fn;
  const Rcpp::Function prior_fn;
  
  ssm_mlg approx_model;
  
  void compute_RR();
  
  arma::vec log_likelihood(
      const unsigned int method, 
      const unsigned int nsim, 
      arma::cube& alpha, 
      arma::mat& weights, 
      arma::umat& indices);
  
  void update_model(const arma::vec& new_theta);
  double log_prior_pdf(const arma::vec& x) const;
  
  // update the approximating Gaussian model
  void approximate();
  void approximate_for_is(const arma::mat& mode_estimate_);
  
  double compute_const_term() const;
  
  // given the mode_estimate, compute y and H of the approximating Gaussian model
  void laplace_iter(const arma::mat& signal);
  
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
  arma::vec importance_weights(const arma::cube& alpha) const;
  // compute unnormalized mode-based scaling terms
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  void update_scales();
  
  // compute logarithms of _unnormalized_ densities g(y_t | alpha_t)
  arma::vec log_obs_density(const unsigned int t, const arma::cube& alpha) const;
};


#endif
