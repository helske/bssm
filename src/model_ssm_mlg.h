// multivariate linear Gaussian state space model

#ifndef SSM_MLG_H
#define SSM_MLG_H

#include "bssm.h"
#include <sitmo.h>

class ssm_mlg {
  
public:
  // constructor from Rcpp::List
  ssm_mlg(
    const Rcpp::List model, 
    const unsigned int seed = 1,
    const double zero_tol = 1e-8);
  
  // constructor from armadillo objects
  ssm_mlg(
    const arma::mat& y, 
    const arma::cube& Z,
    const arma::cube& H,
    const arma::cube& T,
    const arma::cube& R,
    const arma::vec& a1,
    const arma::mat& P1,
    const arma::mat& D,
    const arma::mat& C,
    const arma::mat& xreg, 
    const arma::vec& theta,
    const unsigned int seed,
    const Rcpp::Function update_fn,
    const Rcpp::Function prior_fn,
    const double zero_tol = 1e-8);
  
  arma::mat y;
  arma::cube Z;
  arma::cube H;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::mat D;
  arma::mat C;
  arma::mat xreg;
  
  const unsigned int n; // number of time points
  const unsigned int m; // number of states
  const unsigned int k; // number of etas
  const unsigned int p; // number of series
  
  // is the matrix/vector time-varying?
  const unsigned int Ztv;
  const unsigned int Htv;
  const unsigned int Ttv;
  const unsigned int Rtv;
  const unsigned int Dtv;
  const unsigned int Ctv;
  
  arma::vec theta; 
  
  // random number engine
  sitmo::prng_engine engine;
  // zero-tolerance
  const double zero_tol;
  arma::cube HH;
  arma::cube RR;
  
  // R functions
  const Rcpp::Function update_fn;
  const Rcpp::Function prior_fn;
  
  void compute_RR();
  void compute_HH();
  
  void update_model(const arma::vec& new_theta);
  double log_prior_pdf(const arma::vec& x) const;
  
  // compute the log-likelihood using Kalman filter
  double log_likelihood() const;
  arma::cube simulate_states(const unsigned int nsim);
  
  double filter(arma::mat& at, arma::mat& att, arma::cube& Pt, arma::cube& Ptt) const;
  void smoother(arma::mat& at, arma::cube& Pt) const; 
  // perform fast state smoothing
  arma::mat fast_smoother() const;
  // smoothing which also returns covariances cov(alpha_t, alpha_t-1)
  void smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const;
};


#endif
