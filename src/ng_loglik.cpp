#include "ng_loglik.h"
#include "distr_consts.h"

template double compute_ung_loglik(ung_ssm model, unsigned int simulation_method, unsigned int nsim_states,
  arma::vec mode_estimate, unsigned int max_iter, double conv_tol);
template double compute_ung_loglik(ung_bsm model, unsigned int simulation_method, unsigned int nsim_states,
  arma::vec mode_estimate, unsigned int max_iter, double conv_tol);
template double compute_ung_loglik(ung_ar1 model, unsigned int simulation_method, unsigned int nsim_states,
  arma::vec mode_estimate, unsigned int max_iter, double conv_tol);
template double compute_ung_loglik(ung_svm model, unsigned int simulation_method, unsigned int nsim_states,
  arma::vec mode_estimate, unsigned int max_iter, double conv_tol);

template<class T>
double compute_ung_loglik(T model, const unsigned int simulation_method, 
  const unsigned int nsim_states, arma::vec mode_estimate, 
  const unsigned int max_iter, const double conv_tol) {
  
  double loglik = -1e300;
  
  // bootstrap filter
  if(simulation_method == 2) {
    arma::cube alpha(model.m, model.n + 1, nsim_states);
    arma::mat weights(nsim_states, model.n + 1);
    arma::umat indices(nsim_states, model.n);
    loglik = model.bsf_filter(nsim_states, alpha, weights, indices);
  } else {
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
    
    if(nsim_states > 0) {
      // psi-PF
      if (simulation_method == 1) {
        arma::cube alpha(model.m, model.n + 1, nsim_states);
        arma::mat weights(nsim_states, model.n + 1);
        arma::umat indices(nsim_states, model.n);
        
        loglik =  model.psi_filter(approx_model, approx_loglik, scales, 
          nsim_states, alpha, weights, indices);
      } else {
        //SPDK
        arma::cube alpha = approx_model.simulate_states(nsim_states, true);
        arma::vec weights(nsim_states, arma::fill::zeros);
        for (unsigned int t = 0; t < model.n; t++) {
          weights += model.log_weights(approx_model, t, alpha);
        }
        weights -= arma::accu(scales);
        double maxw = weights.max();
        weights = arma::exp(weights - maxw);
        loglik = approx_loglik + std::log(arma::mean(weights)) + maxw;
      }
    } else {
      loglik = approx_loglik;
    }
  }
  return loglik;
}

