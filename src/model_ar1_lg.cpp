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
  
  
  T(0, 0, 0) = new_theta(0);
  R(0, 0, 0) = new_theta(1);
  RR(0, 0, 0) = std::pow(new_theta(1), 2);
  if (mu_est) {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - new_theta(0)));
  }
  P1(0, 0) = RR(0, 0, 0) / (1.0 - std::pow(new_theta(0), 2));
  
  
  if(sd_y_est) {
    H(0) = new_theta(2 + mu_est);
    HH(0) = std::pow(H(0), 2);
  }
  
  if(xreg.n_cols > 0) {
    beta = new_theta.subvec(new_theta.n_elem - xreg.n_cols, new_theta.n_elem - 1);
    compute_xbeta();
  }
  theta = new_theta;
}
void ar1_lg::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {
  
  // sampling of all parameters is on constrained scale, would make sense to 
  // modify as in bsm models
  T(0, 0, 0) = new_theta(0);
  R(0, 0, 0) = new_theta(1);
  RR(0, 0, 0) = std::pow(new_theta(1), 2);
  if (mu_est) {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - new_theta(0)));
  }
  P1(0, 0) = RR(0, 0, 0) / (1.0 - std::pow(new_theta(0), 2));
  
  if(sd_y_est) {
    H(0) = new_theta(2 + mu_est);
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


