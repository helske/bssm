// store all model components dependent on theta

#ifndef PARSET_GSV_H
#define PARSET_GSV_H

#include "bssm.h"

class ssm_gsv;

class parset_gsv {
  
public:
  parset_gsv(const ssm_gsv& model, const arma::mat& theta, const Rcpp::Function update_fn);
  
  unsigned int n;
  bool est_Z_mu;
  bool est_T_mu;
  bool est_R_mu;
  bool est_a1_mu;
  bool est_P1_mu;
  bool est_C_mu;
  bool est_D_mu;
  
  arma::cube Z_mu;
  arma::field<arma::cube> T_mu;
  arma::field<arma::cube> R_mu;
  arma::mat a1_mu;
  arma::cube P1_mu;
  arma::mat D_mu;
  arma::cube C_mu;
  
  bool est_Z_sv;
  bool est_T_sv;
  bool est_R_sv;
  bool est_a1_sv;
  bool est_P1_sv;
  bool est_C_sv;
  bool est_D_sv;
  
  arma::cube Z_sv;
  arma::field<arma::cube> T_sv;
  arma::field<arma::cube> R_sv;
  arma::mat a1_sv;
  arma::cube P1_sv;
  arma::mat D_sv;
  arma::cube C_sv;
  
  void update(ssm_gsv& model, const unsigned int i);
  
};


#endif
