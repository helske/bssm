#ifndef NG_LOGLIK_H
#define NG_LOGLIK_H

#include "bssm.h"
#include "ung_ssm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "ung_ar1.h"
template<class T>
double compute_ung_loglik(T model, const unsigned int simulation_method, 
  const unsigned int nsim_states, arma::vec mode_estimate, 
  const unsigned int max_iter, const double conv_tol);

#endif
