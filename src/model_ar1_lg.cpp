#include "model_ar1_lg.h"

// from Rcpp::List
ar1_lg::ar1_lg(const Rcpp::List model, const unsigned int seed) :
  ssm_ulg(model, seed), 
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  mu_est(Rcpp::as<bool>(model["mu_est"])), 
  sd_y_est(Rcpp::as<bool>(model["sd_y_est"])) {
}

void ar1_lg::update_model(const arma::vec& new_theta) {
  
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
  
  if(sd_y_est) {
    H(0) = std::exp(new_theta(2 + mu_est));
    HH(0) = std::pow(H(0), 2);
  }
  
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}
void ar1_lg::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
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
  
  if(sd_y_est) {
    H(0) = std::exp(new_theta(2 + mu_est));
    HH(0) = std::pow(H(0), 2);
  }
  
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}

double ar1_lg::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  
  double log_prior = 0.0;
  arma::vec pars = x;
 
  // sigma
  pars(1) = std::exp(pars(1));
  // sd_y
  pars(2 + mu_est) = std::exp(pars(2 + mu_est));
  // add log-jacobians
  log_prior += x(1) + x(2 + mu_est);

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


