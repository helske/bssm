#include "ng_psi_filter.h"

#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "distr_consts.h"

template double compute_ung_psi_filter(ung_ssm model, const unsigned int nsim_states, 
  arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
  arma::cube& alpha, arma::mat& weights, arma::umat& indices);
template double compute_ung_psi_filter(ung_bsm model, const unsigned int nsim_states, 
  arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
  arma::cube& alpha, arma::mat& weights, arma::umat& indices);
template double compute_ung_psi_filter(ung_svm model, const unsigned int nsim_states, 
  arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
  arma::cube& alpha, arma::mat& weights, arma::umat& indices);

template<class T>
double compute_ung_psi_filter(T model, const unsigned int nsim_states, 
  arma::vec mode_estimate, const unsigned int max_iter, const double conv_tol,
  arma::cube& alpha, arma::mat& weights, arma::umat& indices) {
  
  ugg_ssm approx_model = model.approximate(mode_estimate, max_iter, conv_tol);
  // compute the log-likelihood of the approximate model
  
  double gaussian_loglik = approx_model.log_likelihood();
  // compute unnormalized mode-based correction terms 
  // log[g(y_t | ^alpha_t) / ~g(y_t | ^alpha_t)]
  arma::vec scales = model.scaling_factors(approx_model, mode_estimate);
  // compute the constant term
  double const_term = compute_const_term(model, approx_model); 
  // log-likelihood approximation
  double approx_loglik = gaussian_loglik + const_term + arma::accu(scales);
  
  double loglik = model.psi_filter(approx_model, approx_loglik, scales, 
    nsim_states, alpha, weights, indices);
  return loglik;
}

