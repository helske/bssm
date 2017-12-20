#ifndef UNG_SVM_H
#define UNG_SVM_H

#include "ung_ssm.h"

class ung_svm: public ung_ssm {
  
public:

  ung_svm(const Rcpp::List& model, unsigned int seed);

  
  // update model given the parameters theta
  void update_model(const arma::vec& theta);
  double log_prior_pdf(const arma::vec& x) const;
  double log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const;
  
private:
  unsigned int svm_type;
};

#endif
