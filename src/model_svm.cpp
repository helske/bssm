#include "model_svm.h"

// construct SV model from Rcpp::List
svm::svm(const Rcpp::List model, const unsigned int seed) :
  ssm_ung(model, seed), 
  prior_distributions(Rcpp::as<arma::uvec>(model["prior_distributions"])), 
  prior_parameters(Rcpp::as<arma::mat>(model["prior_parameters"])),
  svm_type(model["svm_type"]) {
}

// update model given the parameters theta
void svm::update_model(const arma::vec& new_theta) {
  
  if(svm_type == 0) {
    phi = new_theta(2);
  } else {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - new_theta(0)));
  }
  
  T(0, 0, 0) = new_theta(0);
  R(0, 0, 0) = new_theta(1);
  compute_RR();
  P1(0, 0) = new_theta(1) * new_theta(1) / (1 - new_theta(0) * new_theta(0));
  
  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}
void svm::update_model(const arma::vec& new_theta, const Rcpp::Function update_fn) {

  if(svm_type == 0) {
    phi = new_theta(2);
  } else {
    a1(0) = new_theta(2);
    C.fill(new_theta(2) * (1.0 - new_theta(0)));
  }

  T(0, 0, 0) = new_theta(0);
  R(0, 0, 0) = new_theta(1);
  compute_RR();
  P1(0, 0) = new_theta(1) * new_theta(1) / (1 - new_theta(0) * new_theta(0));

  theta = new_theta;
  // approximation does not match theta anymore (keep as -1 if so)
  if (approx_state > 0) approx_state = 0;
}


double svm::log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const {
  
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

