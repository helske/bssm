// univariate linear Gaussian state space model
#ifndef UGG_SSM_H
#define UGG_SSM_H

#include <sitmo.h>
#include "bssm.h"

class ugg_ssm {
  
public:
  
  // constructor from Rcpp::List
  ugg_ssm(const Rcpp::List& model, 
    const unsigned int seed = 1, 
    const arma::uvec& Z_ind_ = arma::uvec(), 
    const arma::uvec& H_ind_ = arma::uvec(), 
    const arma::uvec& T_ind_ = arma::uvec(), 
    const arma::uvec& R_ind_ = arma::uvec());
  
  // constructor from armadillo objects
  ugg_ssm(const arma::vec& y, const arma::mat& Z, const arma::vec& H, 
    const arma::cube& T, const arma::cube& R, const arma::vec& a1, 
    const arma::mat& P1, const arma::mat& xreg, const arma::vec& beta, 
    const arma::vec& D, const arma::mat& C, 
    const unsigned int seed = 1, 
    const arma::vec& theta = arma::vec(),
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
  
  // compute the log-likelihood
  double log_likelihood() const;
  
  arma::cube simulate_states(const unsigned int nsim_states, 
    const bool use_antithetic = true);
  
  // compute the covariance matrices
  void compute_RR();
  void compute_HH() { HH = square(H); }
  // compute the regression part
  void compute_xbeta() { xbeta = xreg * beta; }
  
  // perform fast state smoothing
  arma::mat fast_smoother() const;
  // fast smoothing using precomputed matrices
  arma::mat fast_smoother(const arma::vec& Ft, const arma::mat& Kt,
    const arma::cube& Lt) const;
  // fast smoothing which returns also Ft, Kt, and Lt
  arma::mat fast_precomputing_smoother(arma::vec& Ft, arma::mat& Kt, 
    arma::cube& Lt) const;
  // smoothing which also returns covariances cov(alpha_t, alpha_t-1)
  void smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const;
  double filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
    arma::cube& Ptt) const;
  void smoother(arma::mat& at, arma::cube& Pt) const;
  double bsf_filter(const unsigned int nsim, arma::cube& alpha,
    arma::mat& weights, arma::umat& indices);
 
  Rcpp::List predict_interval(const arma::vec& probs, const arma::mat& theta,
    const arma::mat& alpha, const arma::uvec& counts, const unsigned int predict_type);
  arma::cube predict_sample(const arma::mat& theta,
    const arma::mat& alpha, const arma::uvec& counts, const unsigned int predict_type
    , const unsigned int nsim);
  arma::mat sample_model(const unsigned int predict_type, const unsigned int nsim);
  
  arma::vec y;
  arma::mat Z;
  arma::vec H;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::mat xreg;
  arma::vec beta;
  arma::vec D;
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
  
  arma::vec HH;
  arma::cube RR;
  arma::vec xbeta;
  sitmo::prng_engine engine;
  const double zero_tol;
  
  arma::vec theta;
  const arma::uvec prior_distributions;
  const arma::mat prior_parameters;

private:
  arma::uvec Z_ind;
  arma::uvec H_ind;
  arma::uvec T_ind;
  arma::uvec R_ind;
};


#endif
