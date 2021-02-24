// store all model components dependent on theta

#ifndef PARSET_LG_H
#define PARSET_LG_H

#include "bssm.h"

class ssm_ulg;
class ssm_mlg;

class parset_ulg {
  
public:
  parset_ulg(const ssm_ulg& model, const arma::mat& theta, const Rcpp::Function update_fn);
  
  unsigned int n;
  bool est_H;
  bool est_Z;
  bool est_T;
  bool est_R;
  bool est_a1;
  bool est_P1;
  bool est_C;
  bool est_D;
  bool est_beta;
  
  
  arma::mat H;
  arma::cube Z;
  arma::field<arma::cube> T;
  arma::field<arma::cube> R;
  arma::mat a1;
  arma::cube P1;
  arma::mat D;
  arma::cube C;
  arma::mat beta;
  
  
  void update(ssm_ulg& model, const unsigned int i);
  
};
class parset_mlg {
  
public:
  parset_mlg(const ssm_mlg& model, const arma::mat& theta, const Rcpp::Function update_fn);
  
  unsigned int n;
  bool est_H;
  bool est_Z;
  bool est_T;
  bool est_R;
  bool est_a1;
  bool est_P1;
  bool est_C;
  bool est_D;
  
  arma::field<arma::cube> H;
  arma::field<arma::cube> Z;
  arma::field<arma::cube> T;
  arma::field<arma::cube> R;
  arma::mat a1;
  arma::cube P1;
  arma::cube D;
  arma::cube C;
  
  void update(ssm_mlg& model, const unsigned int i);
  
};
#endif
