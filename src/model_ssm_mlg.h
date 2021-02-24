// multivariate linear Gaussian state space model

#ifndef SSM_MLG_H
#define SSM_MLG_H

#include "bssm.h"
#include <sitmo.h>

extern Rcpp::Function default_update_fn;

class ssm_mlg {
  
public:
  // constructor from Rcpp::List
  ssm_mlg(
    const Rcpp::List model, 
    const unsigned int seed = 1,
    const double zero_tol = 1e-12);
  
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
    const arma::vec& theta,
    const unsigned int seed,
    const double zero_tol = 1e-12);
  
  arma::mat y;
  arma::cube Z;
  arma::cube H;
  arma::cube T;
  arma::cube R;
  arma::vec a1;
  arma::mat P1;
  arma::mat D;
  arma::mat C;
  
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
  
  void compute_RR(){
    for (unsigned int t = 0; t < R.n_slices; t++) {
      RR.slice(t) = R.slice(t) * R.slice(t).t();
    }
  }
  
  void compute_HH(){
    for (unsigned int t = 0; t < H.n_slices; t++) {
      HH.slice(t) = H.slice(t) * H.slice(t).t();
    }
  }
  
  void update_model(const arma::vec& new_theta, const Rcpp::Function update_fn);
  double log_prior_pdf(const arma::vec& x, const Rcpp::Function prior_fn) const;
  
  // compute the log-likelihood using Kalman filter
  double log_likelihood() const;
  arma::cube simulate_states(const unsigned int nsim);
  
  double filter(arma::mat& at, arma::mat& att, arma::cube& Pt, arma::cube& Ptt) const;
  
  void psi_filter(const unsigned int nsim, arma::cube& alpha);
    
  void smoother(arma::mat& at, arma::cube& Pt) const; 
  // perform fast state smoothing
  arma::mat fast_smoother() const;
  // smoothing which also returns covariances cov(alpha_t, alpha_t-1)
  void smoother_ccov(arma::mat& at, arma::cube& Pt, arma::cube& ccov) const;
  
  arma::cube predict_sample(const arma::mat& theta_posterior,
    const arma::mat& alpha, const unsigned int predict_type, 
    const Rcpp::Function update_fn = default_update_fn);
  
  arma::mat sample_model(const unsigned int predict_type);
  
  arma::cube predict_past(const arma::mat& theta_posterior,
    const arma::cube& alpha, const unsigned int predict_type, 
    const Rcpp::Function update_fn = default_update_fn);
};


#endif
