#ifndef SVM_H
#define SVM_H

#include "ngssm.h"

class svm: public ngssm {

public:

  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, arma::mat, arma::vec, unsigned int);

  svm(arma::vec, arma::mat, arma::cube, arma::cube, arma::vec,
    arma::mat, arma::vec, arma::mat, arma::vec, unsigned int, bool);

  double gap_filter(unsigned int, arma::cube&, arma::vec&, arma::vec&);
  double proposal(const arma::vec&, const arma::vec&);
  void update_model(arma::vec);
  arma::vec get_theta(void);
  arma::vec approx_iter(arma::vec&);
  double logp_y(arma::vec&);
  arma::vec nz_y;
  const bool log_space;
};

#endif
