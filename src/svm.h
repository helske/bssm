#ifndef SVM_H
#define SVM_H

#include "ngssm.h"

class svm: public ngssm {

public:


  svm(const List&, unsigned int);

  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, double, arma::mat, arma::vec, arma::mat, unsigned int, unsigned int);

  void update_model(arma::vec);
  arma::vec get_theta(void);
  arma::vec approx_iter(arma::vec&);

  arma::vec nz_y;
  unsigned int svm_type;
};

#endif
