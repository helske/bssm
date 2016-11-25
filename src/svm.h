#ifndef SVM_H
#define SVM_H

#include "ngssm.h"

class svm: public ngssm {
  
public:
  
  svm(const Rcpp::List&, unsigned int);
  svm(const Rcpp::List&, unsigned int, bool);
  
  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, double, arma::mat, arma::vec, arma::mat, unsigned int, unsigned int);

  void update_model(arma::vec);
  arma::vec get_theta(void);
  arma::vec approx_iter(arma::vec&);

  double prior_pdf(const arma::vec&, const arma::uvec&, const arma::mat&);
  
  arma::vec nz_y;
  unsigned int svm_type;
  bool gkl;
};

#endif
