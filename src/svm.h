#ifndef SVM_H
#define SVM_H

#include "ngssm.h"

class svm: public ngssm {

public:

  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, arma::mat, arma::vec, unsigned int);

  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, arma::mat, arma::vec, unsigned int, double);

  double particle_filter2(unsigned int, arma::cube&, arma::mat&, arma::umat& ind, arma::vec, double);
  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);
  arma::vec approx_iter(arma::vec&);
  double logp_y(arma::vec&);
  arma::vec nz_y;
  double prior_sd;
};

#endif