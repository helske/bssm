#include "milstein_functions.h"

// [[Rcpp::export]]
double R_milstein(const double x0, const unsigned int L, const double t,
  const arma::vec& theta,
  SEXP drift_pntr, SEXP diffusion_pntr, SEXP ddiffusion_pntr,
  bool positive, const unsigned int seed) {

  sitmo::prng_engine eng(seed);

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  fnPtr drift = *xpfun_drift;

  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  fnPtr diffusion= *xpfun_diffusion;

  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  fnPtr ddiffusion = *xpfun_ddiffusion;

  return milstein(x0, L, t, theta,
    drift, diffusion, ddiffusion, positive, eng);
}

// [[Rcpp::export]]
double R_milstein_joint(const double x0,
  const unsigned int L_c, const unsigned int L_f, const double t,
  const arma::vec& theta,
  SEXP drift_pntr, SEXP diffusion_pntr, SEXP ddiffusion_pntr,
  bool positive, const unsigned int seed) {

  sitmo::prng_engine eng_c(seed);
  sitmo::prng_engine eng_f(seed + 1);

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  fnPtr drift = *xpfun_drift;

  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  fnPtr diffusion= *xpfun_diffusion;

  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  fnPtr ddiffusion = *xpfun_ddiffusion;

  return milstein_joint(x0, L_c, L_f, t, theta,
    drift, diffusion, ddiffusion,
    positive, eng_c, eng_f);
}
