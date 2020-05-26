#include "ung_ar1.h"

// from Rcpp::List
ung_ar1::ung_ar1(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), mu_est(Rcpp::as<bool>(model["mu_est"])) {
}

void ung_ar1::update_model(const arma::vec& new_theta) {
  
  
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
}

double ung_ar1::log_prior_pdf(const arma::vec& x) const {
  
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
    }
  }
  return log_prior;
}

