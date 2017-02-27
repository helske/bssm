#ifndef UNG_SVM_H
#define UNG_SVM_H

#include "ung_ssm.h"

class ung_svm: public ung_ssm {
  
public:

  ung_svm(const Rcpp::List& model, unsigned int seed);

  
  // update model given the parameters theta
  void set_theta(const arma::vec& theta);
  // extract theta from the model
  arma::vec get_theta() const;
  
private:
  unsigned int svm_type;
};

#endif
