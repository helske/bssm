#ifndef SVM_H
#define SVM_H

#include "model_ssm_ung.h"


extern Rcpp::Function default_prior_fn;

class svm: public ssm_ung {
  
public:

  svm(const Rcpp::List model, unsigned int seed = 1);

  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn = default_prior_fn) const;  

private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  unsigned int svm_type;
};

#endif
