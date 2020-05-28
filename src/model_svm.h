#ifndef svm_H
#define svm_H

#include "model_ssm_ung.h"

class svm: public ssm_ung {
  
public:

  svm(const Rcpp::List& model, unsigned int seed = 1);

  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  double log_prior_pdf(const arma::vec& x);  

private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  unsigned int svm_type;
};

#endif
