#include "model_ar1_ng.h"

// from Rcpp::List
ar1_ng::ar1_ng(const Rcpp::List model, const unsigned int seed) :
  ssm_ung(model, seed), 
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  mu_est(Rcpp::as<bool>(model["mu_est"])), 
  phi_est(Rcpp::as<bool>(model["phi_est"])) {
}

void ar1_ng::update_model(const arma::vec& new_theta) {
  
  T(0, 0, 0) = new_theta(0);
  R(0, 0, 0) = new_theta(1);
  if (mu_est) {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - new_theta(0)));
  }
  P1(0, 0) = std::pow(new_theta(1), 2) / (1.0 - std::pow(new_theta(0), 2));
  
  compute_RR();
  
  if(phi_est) {
    phi = new_theta(2 + mu_est);
  }
  
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;  
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

double ar1_ng::log_prior_pdf(const arma::vec& x) const {
  
  double log_prior = 0.0;
  
  for(unsigned int i = 0; i < x.n_elem; i++) {
    switch(prior_distributions(i)) {
    case 0  :
      if (x(i) < prior_parameters(0, i) || x(i) > prior_parameters(1, i)) {
        return -std::numeric_limits<double>::infinity(); 
      }
      break;
    case 1  :
      if (x(i) < 0) {
        return -std::numeric_limits<double>::infinity();
      } else {
        log_prior -= 0.5 * std::pow(x(i) / prior_parameters(0, i), 2);
      }
      break;
    case 2  :
      log_prior -= 0.5 * std::pow((x(i) - prior_parameters(0, i)) / prior_parameters(1, i), 2);
      break;
    case 3 : // truncated normal
      if (x(i) < prior_parameters(2, i) || x(i) > prior_parameters(3, i)) {
        return -std::numeric_limits<double>::infinity(); 
      } else {
        log_prior -= 0.5 * std::pow((x(i) - prior_parameters(0, i)) / prior_parameters(1, i), 2);
      }
      break;
    case 4 : // gamma
      if (x(i) < 0) {
        return -std::numeric_limits<double>::infinity(); 
      } else {
        log_prior += (prior_parameters(0, i) - 1) * log(x(i)) - prior_parameters(1, i) * x(i);
        
      }
      break;
    }
  }
  return log_prior;
}

