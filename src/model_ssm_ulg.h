// univariate linear Gaussian state space model

#ifndef SSM_ULG_H
#define SSM_ULG_H


#include "bssm.h"
#include <sitmo.h>

extern Rcpp::Function default_update_fn;

class ssm_ulg {
  
public:
  
  // constructor from Rcpp::List
  ssm_ulg(const Rcpp::List model, 
    const unsigned int seed = 1,
    const double zero_tol = 1e-12);
  
  // constructor from armadillo objects
  ssm_ulg(const arma::vec& y, 
    const arma::mat& Z, 
    const arma::vec& H, 
    const arma::cube& T, 
    const arma::cube& R, 
    const arma::vec& a1, 
    const arma::mat& P1, 
    const arma::vec& D, 
    const arma::mat& C, 
    const arma::mat& xreg, 
    const arma::vec& beta, 
    const arma::vec& theta,
    const unsigned int seed,
    const double zero_tol = 1e-12);
  
  arma::vec y;
  arma::mat Z;
  arma::vec H;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::vec D;
  arma::mat C;
  arma::mat xreg;
  arma::vec beta; 
  
  const unsigned int n; // number of time points
  const unsigned int m; // number of states
  const unsigned int k; // number of etas
  static const unsigned int p = 1; // number of series
  
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
  
  arma::vec HH;
  arma::cube RR;
  arma::vec xbeta;
  
  virtual void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  virtual double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const;
  
  void compute_RR(){
    for (unsigned int t = 0; t < R.n_slices; t++) {
      RR.slice(t) = R.slice(t) * R.slice(t).t();
    }
  }
  
  void compute_HH() { HH = arma::square(H); }
  void compute_xbeta() { xbeta = xreg * beta; }
  
  // compute the log-likelihood
  double log_likelihood() const;
  arma::cube simulate_states(const unsigned int nsim, 
    const bool use_antithetic = true);
  
  double filter(arma::mat& at, arma::mat& att, arma::cube& Pt,
    arma::cube& Ptt) const;
  void smoother(arma::mat& at, arma::cube& Pt) const;
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

  double bsf_filter(const unsigned int nsim, arma::cube& alpha,
    arma::mat& weights, arma::umat& indices);
  // simulation smoothing using twisted smc
  void psi_filter(const unsigned int nsim, arma::cube& alpha);
  
  arma::cube predict_sample(const arma::mat& theta_posterior,
    const arma::mat& alpha, const unsigned int predict_type, 
    const Rcpp::Function update_fn = default_update_fn);
  
  arma::mat sample_model(const unsigned int predict_type);
  
  arma::cube predict_past(const arma::mat& theta_posterior,
    const arma::cube& alpha, const unsigned int predict_type, 
    const Rcpp::Function update_fn = default_update_fn);
  
};


#endif
