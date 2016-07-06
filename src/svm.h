#ifndef SVM_H
#define SVM_H

#include "ngssm.h"

class svm: public ngssm {

public:

  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, double, double, unsigned int);

  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);
  arma::vec approx_iter(arma::vec&);
  double logp_y(arma::vec&);
  double mean;
  double sigma;
  arma::vec nz_y;
};

#endif
