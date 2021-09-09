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
  
  double rho = new_theta(0);
  double sigma = std::exp(new_theta(1));
  T(0, 0, 0) = rho;
  R(0, 0, 0) = sigma;
  RR(0, 0, 0) = std::pow(sigma, 2);
  if (mu_est) {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - rho));
  }
  P1(0, 0) = RR(0, 0, 0) / (1.0 - std::pow(rho, 2));
  
  if(phi_est) {
    phi = std::exp(new_theta(2 + mu_est));
  }
  
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;  
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}
void ar1_ng::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  double rho = new_theta(0);
  double sigma = std::exp(new_theta(1));
  T(0, 0, 0) = rho;
  R(0, 0, 0) = sigma;
  RR(0, 0, 0) = std::pow(sigma, 2);
  if (mu_est) {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - rho));
  }
  P1(0, 0) = RR(0, 0, 0) / (1.0 - std::pow(rho, 2));
  
  if(phi_est) {
    phi = std::exp(new_theta(2 + mu_est));
  }
  
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;  
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}

double ar1_ng::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  
  double log_prior = 0.0;
  arma::vec pars = x;

  // sigma
  pars(1) = std::exp(pars(1));
  // add log-jacobian
  log_prior += x(1);
  // phi
  if (phi_est) {
    pars(2 + mu_est) = std::exp(pars(2 + mu_est));
    log_prior += x(2 + mu_est);
  }
  
  for(unsigned int i = 0; i < pars.n_elem; i++) {
    switch(prior_distributions(i)) {
    case 0  :
      if (pars(i) < prior_parameters(0, i) || pars(i) > prior_parameters(1, i)) {
        return -std::numeric_limits<double>::infinity(); 
      }
      break;
    case 1  :
      if (pars(i) < 0) {
        return -std::numeric_limits<double>::infinity();
      } else {
        log_prior -= 0.5 * std::pow(pars(i) / prior_parameters(0, i), 2);
      }
      break;
    case 2  :
      log_prior -= 0.5 * std::pow((pars(i) - prior_parameters(0, i)) / 
        prior_parameters(1, i), 2);
      break;
    case 3 : // truncated normal
      if (pars(i) < prior_parameters(2, i) || pars(i) > prior_parameters(3, i)) {
        return -std::numeric_limits<double>::infinity(); 
      } else {
        log_prior -= 0.5 * std::pow((pars(i) - prior_parameters(0, i)) / 
          prior_parameters(1, i), 2);
      }
      break;
    case 4 : // gamma
      if (pars(i) < 0) {
        return -std::numeric_limits<double>::infinity(); 
      } else {
        log_prior += (prior_parameters(0, i) - 1) * 
          log(pars(i)) - prior_parameters(1, i) * pars(i);
        
      }
      break;
    }
  }
  return log_prior;
}

