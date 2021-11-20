#include "model_ssm_sde.h"

#include "filter_smoother.h"
#include "summary.h"
#include "mcmc.h"
#include "approx_mcmc.h"


// [[Rcpp::export]]
double loglik_sde(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const arma::vec& theta, const unsigned int nsim,
  const unsigned int L, const unsigned int seed) {


  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  ssm_sde model(y, theta, x0, positive,*xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior,
     L, L, seed);

  unsigned int n = model.n;
  arma::cube alpha(1, n + 1, nsim);
  arma::mat weights(nsim, n + 1);
  arma::umat indices(nsim, n);
  return model.bsf_filter(nsim, L, alpha, weights, indices);
}

// [[Rcpp::export]]
Rcpp::List bsf_sde(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const arma::vec& theta, const unsigned int nsim,
  const unsigned int L, const unsigned int seed) {

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  ssm_sde model(y, theta, x0, positive,*xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior,
     L, L, seed);

  unsigned int n = model.n;
  arma::cube alpha(1, n + 1, nsim, arma::fill::zeros);
  arma::mat weights(nsim, n + 1, arma::fill::zeros);
  arma::umat indices(nsim, n, arma::fill::zeros);
  double loglik = model.bsf_filter(nsim, L, alpha, weights, indices);
  if (!std::isfinite(loglik)) 
    Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
  
  arma::mat at(1, n + 1);
  arma::mat att(1, n + 1);
  arma::cube Pt(1, 1, n + 1);
  arma::cube Ptt(1, 1, n + 1);
  filter_summary(alpha, at, att, Pt, Ptt, weights);

  arma::inplace_trans(at);
  arma::inplace_trans(att);
  return Rcpp::List::create(
    Rcpp::Named("at") = at, Rcpp::Named("att") = att,
    Rcpp::Named("Pt") = Pt, Rcpp::Named("Ptt") = Ptt,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
}

// [[Rcpp::export]]
Rcpp::List bsf_smoother_sde(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const arma::vec& theta, const unsigned int nsim,
  const unsigned int L, const unsigned int seed) {

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  ssm_sde model(y, theta, x0, positive,*xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior,
     L, L, seed);

  unsigned int n = model.n;
  arma::cube alpha(1, n + 1, nsim, arma::fill::zeros);
  arma::mat weights(nsim, n + 1, arma::fill::zeros);
  arma::umat indices(nsim, n, arma::fill::zeros);
  double loglik = model.bsf_filter(nsim, L, alpha, weights, indices);
  if (!std::isfinite(loglik)) 
    Rcpp::warning("Particle filtering stopped prematurely due to nonfinite log-likelihood.");
  
  arma::mat alphahat(1, n + 1);
  arma::cube Vt(1, 1, n + 1);

  filter_smoother(alpha, indices);
  summary(alpha, alphahat, Vt); // weights are uniform due to extra time point

  arma::inplace_trans(alphahat);

  return Rcpp::List::create(
    Rcpp::Named("alphahat") = alphahat, Rcpp::Named("Vt") = Vt,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("logLik") = loglik, Rcpp::Named("alpha") = alpha);
}

// [[Rcpp::export]]
Rcpp::List sde_pm_mcmc(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const arma::vec& theta, const unsigned int nsim,
  const unsigned int L,
  const unsigned int seed, const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int type, const bool verbose) {

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  ssm_sde model(y, theta, x0, positive,*xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior,
     L, L, seed);

  mcmc mcmc_run(iter, burnin,
    thin, model.n, 1, target_acceptance, gamma, S, type, verbose);

  mcmc_run.pm_mcmc(model, nsim, end_ram);

  switch (type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("error") = "error");
}

// [[Rcpp::export]]
Rcpp::List sde_da_mcmc(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const arma::vec& theta, const unsigned int nsim,
  const unsigned int L_c, const unsigned int L_f, const unsigned int seed,
  const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int type, const bool verbose) {

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  ssm_sde model(y, theta, x0, positive,*xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior,
     L_f, L_c, seed);

  mcmc mcmc_run(iter, burnin,
    thin, model.n, 1, target_acceptance, gamma, S, type, verbose);

  mcmc_run.da_mcmc(model, nsim, end_ram);

  switch (type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }
  return Rcpp::List::create(Rcpp::Named("error") = "error");
}

// [[Rcpp::export]]
Rcpp::List sde_is_mcmc(const arma::vec& y, const double x0,
  const bool positive, SEXP drift_pntr, SEXP diffusion_pntr,
  SEXP ddiffusion_pntr, SEXP log_prior_pdf_pntr, SEXP log_obs_density_pntr,
  const arma::vec& theta, const unsigned int nsim,
  const unsigned int L_c, const unsigned int L_f, const unsigned int seed,
  const unsigned int iter,
  const unsigned int burnin, const unsigned int thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int is_type, const unsigned int n_threads,
  const unsigned int type, const bool verbose) {

  Rcpp::XPtr<fnPtr> xpfun_drift(drift_pntr);
  Rcpp::XPtr<fnPtr> xpfun_diffusion(diffusion_pntr);
  Rcpp::XPtr<fnPtr> xpfun_ddiffusion(ddiffusion_pntr);
  Rcpp::XPtr<prior_fnPtr> xpfun_prior(log_prior_pdf_pntr);
  Rcpp::XPtr<obs_fnPtr> xpfun_obs(log_obs_density_pntr);

  ssm_sde model(y, theta, x0, positive,*xpfun_drift,
    *xpfun_diffusion, *xpfun_ddiffusion, *xpfun_obs, *xpfun_prior,
    L_f, L_c, seed);

  approx_mcmc mcmc_run(iter, burnin, thin, model.n, 1, 1,
    target_acceptance, gamma, S, type, false, verbose);

  mcmc_run.amcmc(model, nsim, end_ram);

  if(is_type == 3) {
    mcmc_run.expand();
  }

  mcmc_run.is_correction_bsf(model, nsim, is_type, n_threads);

  switch (type) {
  case 1: {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 2: {
    return Rcpp::List::create(
      Rcpp::Named("alphahat") = mcmc_run.alphahat.t(), Rcpp::Named("Vt") = mcmc_run.Vt,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  case 3: {
    return Rcpp::List::create(
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("weights") = mcmc_run.weight_storage,
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("error") = "error");
}
