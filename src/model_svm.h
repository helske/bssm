#ifndef SVM_H
#define SVM_H

#include "model_ssm_ung.h"

class svm: public ssm_ung {
  
public:

  svm(const Rcpp::List model, unsigned int seed = 1);

  // update model given the parameters theta
  void update_model(const arma::vec& new_theta);
  double log_prior_pdf(const arma::vec& x) const;  

private:
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  unsigned int svm_type;
};

#endif
