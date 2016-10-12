#ifndef SVM_H
#define SVM_H

#include "ngssm.h"

class svm: public ngssm {
  
public:
  
  svm(const List&, unsigned int);
  svm(const List&, unsigned int, bool);
  
  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, double, arma::mat, arma::vec, unsigned int);
  
  void update_model(arma::vec);
  arma::vec get_theta(void);
  arma::vec approx_iter(arma::vec&);
  
  double logp_y(arma::vec&);
  double prior_pdf(const arma::vec&, const arma::uvec&, const arma::mat&);
    
  arma::vec nz_y;
  bool gkl;
};

#endif
