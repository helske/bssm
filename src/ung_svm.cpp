#include "ung_svm.h"

// construct SV model from Rcpp::List
ung_svm::ung_svm(const Rcpp::List& model, const unsigned int seed) :
  ung_ssm(model, seed), svm_type(model["svm_type"]) {
}

// update model given the parameters theta
void ung_svm::update_model(const arma::vec& theta) {

  if(svm_type == 0) {
    phi = theta(2);
  } else {
    a1(0) = theta(2);
    C.fill(theta(2) * (1.0 - theta(0)));
  }

  T(0, 0, 0) = theta(0);
  R(0, 0, 0) = theta(1);
  compute_RR();
  P1(0, 0) = theta(1) * theta(1) / (1 - theta(0) * theta(0));

  if(xreg.n_cols > 0) {
    beta = theta.subvec(theta.n_elem - xreg.n_cols, theta.n_elem - 1);
    compute_xbeta();
  }
}
double ung_svm::log_prior_pdf(const arma::vec& x) const {
  
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
    }
  }
  return log_prior;
}

double ung_svm::log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const {
  return 0.0;
}
