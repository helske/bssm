#ifndef NG_LOGLIK_H
#define NG_LOGLIK_H

#include <RcppArmadillo.h>
#include "ung_ssm.h"
#include "ung_bsm.h"

template<class T>
double compute_ung_loglik(T model, unsigned int simulation_method, unsigned int nsim_states,
  arma::vec mode_estimate, unsigned int max_iter, double conv_tol);

#endif
