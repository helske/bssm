// univariate linear Gaussian state space model
#ifndef UGG_SSM_H
#define UGG_SSM_H

#include <RcppArmadillo.h>
#include "gg_abc.h"

class ugg_ssm : public gg_abc {
  
public:
  
  // constructor from Rcpp::List
  ugg_ssm(const Rcpp::List& model, 
    const unsigned int seed = 1, 
    const arma::uvec& Z_ind = arma::uvec(), 
    const arma::uvec& H_ind = arma::uvec(), 
    const arma::uvec& T_ind = arma::uvec(), 
    const arma::uvec& R_ind = arma::uvec());
  
  // constructor from armadillo objects
  ugg_ssm(const arma::vec& y, const arma::mat& Z, const arma::vec& H, 
    const arma::cube& T, const arma::cube& R, const arma::vec& a1, 
    const arma::mat& P1, const arma::mat& xreg, const arma::vec& beta, 
    const arma::vec& D, const arma::mat& C, const unsigned int seed = 1, 
    const arma::uvec& Z_ind = arma::uvec(), 
    const arma::uvec& H_ind = arma::uvec(), 
    const arma::uvec& T_ind = arma::uvec(), 
    const arma::uvec& R_ind = arma::uvec());
  
  // update model matrices
  void set_theta(const arma::vec& theta);
  // get current value of theta
  arma::vec get_theta() const;
  
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
    const arma::mat& alpha, const arma::uvec& counts, const bool predict_obs);
  arma::cube predict_sample(const arma::mat& theta,
    const arma::mat& alpha, const arma::uvec& counts, const bool predict_obs);
  arma::mat sample_model(const bool simulate_obs);
  
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
  
  std::mt19937 engine;
  const double zero_tol;
  
private:
  arma::uvec Z_ind;
  arma::uvec H_ind;
  arma::uvec T_ind;
  arma::uvec R_ind;
  unsigned int seed;
};


#endif
