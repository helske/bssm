// store all model components dependent on theta

#ifndef PARSET_NG_H
#define PARSET_NG_H

#include "bssm.h"

class ssm_ung;
class ssm_mng;

class parset_ung {
  
public:
  parset_ung(const ssm_ung& model, const arma::mat& theta, const Rcpp::Function update_fn);
  
  unsigned int n;
  bool est_phi;
  bool est_Z;
  bool est_T;
  bool est_R;
  bool est_a1;
  bool est_P1;
  bool est_C;
  bool est_D;
  bool est_beta;
  
  arma::vec phi;
  arma::cube Z;
  arma::field<arma::cube> T;
  arma::field<arma::cube> R;
  arma::mat a1;
  arma::cube P1;
  arma::mat D;
  arma::cube C;
  arma::mat beta;
  
  void update(ssm_ung& model, const unsigned int i);
  
};

class parset_mng {
  
public:
  parset_mng(const ssm_mng& model, const arma::mat& theta, const Rcpp::Function update_fn);
  
  unsigned int n;
  bool est_phi;
  bool est_Z;
  bool est_T;
  bool est_R;
  bool est_a1;
  bool est_P1;
  bool est_C;
  bool est_D;
  
  arma::mat phi;
  arma::field<arma::cube> Z;
  arma::field<arma::cube> T;
  arma::field<arma::cube> R;
  arma::mat a1;
  arma::cube P1;
  arma::cube D;
  arma::cube C;
  
  void update(ssm_mng& model, const unsigned int i);
};


#endif
