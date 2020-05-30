#include "model_sde_ssm.h"

#include "sitmo.h"
#include "filter_smoother.h"
#include "summary.h"

// [[Rcpp::export]]
Rcpp::List sde_state_sampler_bsf_is2(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const unsigned int nsim,
  const unsigned int L_f, const unsigned int seed,
  const arma::vec& approx_loglik_storage, const arma::mat& theta) {

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  sde_ssm model(y, theta.col(0), x0, positive, *xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior, seed);

  arma::vec weights(theta.n_cols);
  arma::cube alpha(model.n + 1, 1, theta.n_cols);

  for (unsigned int i = 0; i < theta.n_cols; i++) {

    model.theta = theta.col(i);

    arma::cube alpha_i(1, model.n + 1, nsim);
    arma::mat weights_i(nsim, model.n + 1);
    arma::umat indices(nsim, model.n);
    double loglik = model.bsf_filter(nsim, L_f, alpha_i, weights_i, indices);
    // if(arma::is_finite(loglik)) {
      weights(i) = std::exp(loglik - approx_loglik_storage(i));

      filter_smoother(alpha_i, indices);
      arma::vec w = weights_i.col(model.n);
      std::discrete_distribution<unsigned int> sample(w.begin(), w.end());
      alpha.slice(i) = alpha_i.slice(sample(model.engine)).t();
    // } else {
    //   weights(i) = 0.0;
    //   alpha.slice(i).zeros();
    // }
  }
  return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
    Rcpp::Named("weights") = weights);

}
