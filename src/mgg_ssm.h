// multivariate linear Gaussian state space model
#ifndef MGG_H
#define MGG_H

#include <sitmo.h>

#include "bssm.h"

class mgg_ssm {
  
public:
  
  // constructor from Rcpp::List
  mgg_ssm(const Rcpp::List& model, 
    const unsigned int seed = 1, 
    const arma::uvec& Z_ind_ = arma::uvec(), 
    const arma::uvec& H_ind_ = arma::uvec(), 
    const arma::uvec& T_ind_ = arma::uvec(), 
    const arma::uvec& R_ind_ = arma::uvec());
  
  // constructor from armadillo objects
  mgg_ssm(const arma::mat& y, const arma::cube& Z, const arma::cube& H, 
    const arma::cube& T, const arma::cube& R, const arma::vec& a1, 
    const arma::mat& P1, const arma::cube& xreg, const arma::mat& beta, 
    const arma::mat& D, const arma::mat& C, 
    const unsigned int seed = 1, const arma::vec& theta = arma::vec(),
    const arma::uvec& prior_distributions = arma::uvec(),
    const arma::mat& prior_parameters = arma::mat(), 
    const arma::uvec& Z_ind_ = arma::uvec(), 
    const arma::uvec& H_ind_ = arma::uvec(), 
    const arma::uvec& T_ind_ = arma::uvec(), 
    const arma::uvec& R_ind_ = arma::uvec());
  
  // update model matrices
  virtual void update_model(const arma::vec& new_theta);
  virtual double log_prior_pdf(const arma::vec& x) const;
  virtual double log_proposal_ratio(const arma::vec& new_theta, const arma::vec& old_theta) const;
  
  // compute the covariance matrices
  void compute_RR();
  void compute_HH();
  // compute the regression part
  void compute_xbeta();
  
  
  // compute the log-likelihood using Kalman filter
  double log_likelihood() const;
  
  arma::cube simulate_states();
  void smoother(arma::mat& at, arma::cube& Pt) const; 
  // perform fast state smoothing
  arma::mat fast_smoother() const;
  // smoothing which also returns covariances cov(alpha_t, alpha_t-1)
  void smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const;
  double filter(arma::mat& at, arma::mat& att, arma::cube& Pt, arma::cube& Ptt) const;
  arma::mat y;
  arma::cube Z;
  arma::cube H;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::cube xreg;
  arma::mat beta;
  arma::mat D;
  arma::mat C;
  
  const unsigned int Ztv;
  const unsigned int Htv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Dtv;
  const unsigned int Ctv;
  
  const unsigned int n;
  const unsigned int m;
  const unsigned int k;
  const unsigned int p;
  
  arma::cube HH;
  arma::cube RR;
  arma::mat xbeta;
  arma::vec theta;
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;
  
  sitmo::prng_engine engine;
  const double zero_tol;
  
private:
  arma::uvec Z_ind;
  arma::uvec H_ind;
  arma::uvec T_ind;
  arma::uvec R_ind;
};


#endif
