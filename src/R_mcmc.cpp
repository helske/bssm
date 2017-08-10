#include "mcmc.h"
#include "ung_amcmc.h"
#include "nlg_amcmc.h"
#include "ugg_bsm.h"
#include "ung_bsm.h"
#include "ung_svm.h"
#include "nlg_ssm.h"
#include "lgg_ssm.h"

// [[Rcpp::export]]
Rcpp::List gaussian_mcmc(const Rcpp::List& model_,
  const arma::uvec prior_types, const arma::mat prior_pars,
  const bool sim_states, const unsigned int n_iter, const unsigned int n_burnin,
  const unsigned int n_thin, const double gamma, const double target_acceptance,
  const arma::mat S, const unsigned int seed, const bool end_ram,
  const unsigned int n_threads, const int model_type, const arma::uvec& Z_ind,
  const arma::uvec& H_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, sim_states);

  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), seed, Z_ind, H_ind, T_ind, R_ind);
    mcmc_run.mcmc_gaussian(model, end_ram);
    if(sim_states) mcmc_run.state_posterior(model, n_threads);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    if(sim_states) mcmc_run.state_posterior(model, n_threads);
  } break;
  }

  if(sim_states) {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } else {
    return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  }
}
// [[Rcpp::export]]
Rcpp::List gaussian_mcmc_summary(const Rcpp::List& model_,
  const arma::uvec prior_types, const arma::mat prior_pars,
  const unsigned int n_iter, const unsigned int n_burnin,
  const unsigned int n_thin, const double gamma, const double target_acceptance,
  const arma::mat S, const unsigned int seed, const bool end_ram,
  const unsigned int n_threads, const int model_type, const arma::uvec& Z_ind,
  const arma::uvec& H_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, false);

  arma::mat alphahat(m, n);
  arma::cube Vt(m, m, n);

  switch (model_type) {
  case 1: {
    ugg_ssm model(clone(model_), seed, Z_ind, H_ind, T_ind, R_ind);
    mcmc_run.mcmc_gaussian(model, end_ram);
    mcmc_run.state_summary(model, alphahat, Vt);
  } break;
  case 2: {
    ugg_bsm model(clone(model_), seed);
    mcmc_run.mcmc_gaussian(model, end_ram);
    mcmc_run.state_summary(model, alphahat, Vt);
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("alphahat") = alphahat.t(), Rcpp::Named("Vt") = Vt,
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}
// [[Rcpp::export]]
Rcpp::List nongaussian_pm_mcmc(const Rcpp::List& model_,
  const arma::uvec prior_types, const arma::mat prior_pars,
  const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const unsigned int seed, const bool end_ram, const unsigned int n_threads,
  const bool local_approx, const arma::vec initial_mode,
  const unsigned int max_iter, const double conv_tol,
  const unsigned int simulation_method, const int model_type,
  const arma::uvec& Z_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, true);

  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed, Z_ind, T_ind, R_ind);
    switch (simulation_method) {
    case 1:
      mcmc_run.pm_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 2:
      mcmc_run.pm_mcmc_bsf(model, end_ram, nsim_states);
      break;
    case 3:
      mcmc_run.pm_mcmc_spdk(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    }
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    switch (simulation_method) {
    case 1:
      mcmc_run.pm_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 2:
      mcmc_run.pm_mcmc_bsf(model, end_ram, nsim_states);
      break;
    case 3:
      mcmc_run.pm_mcmc_spdk(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    }
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    switch (simulation_method) {
    case 1:
      mcmc_run.pm_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 2:
      mcmc_run.pm_mcmc_bsf(model, end_ram, nsim_states);
      break;
    case 3:
      mcmc_run.pm_mcmc_spdk(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    }
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}


// [[Rcpp::export]]
Rcpp::List nongaussian_da_mcmc(const Rcpp::List& model_,
  const arma::uvec prior_types, const arma::mat prior_pars,
  const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin, const double gamma,
  const double target_acceptance, const arma::mat S, const unsigned int seed,
  const bool end_ram, const unsigned int n_threads, const bool local_approx,
  const arma::vec initial_mode, const unsigned int max_iter, const double conv_tol,
  const unsigned int simulation_method, const int model_type,
  const arma::uvec& Z_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  mcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, true);

  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed, Z_ind, T_ind, R_ind);
    switch (simulation_method) {
    case 1:
      mcmc_run.da_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 2:
      mcmc_run.da_mcmc_bsf(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 3:
      mcmc_run.da_mcmc_spdk(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    }
  } break;
  case 2: {

    ung_bsm model(clone(model_), seed);
    switch (simulation_method) {
    case 1:
      mcmc_run.da_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 2:
      mcmc_run.da_mcmc_bsf(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 3:
      mcmc_run.da_mcmc_spdk(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    }
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    switch (simulation_method) {
    case 1:
      mcmc_run.da_mcmc_psi(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 2:
      mcmc_run.da_mcmc_bsf(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    case 3:
      mcmc_run.da_mcmc_spdk(model, end_ram, nsim_states, local_approx, initial_mode,
        max_iter, conv_tol);
      break;
    }
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}



// [[Rcpp::export]]
Rcpp::List nongaussian_is_mcmc(const Rcpp::List& model_,
  const arma::uvec prior_types, const arma::mat prior_pars,
  const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin, const  double gamma,
  const double target_acceptance, const arma::mat S, const unsigned int seed,
  const bool end_ram, const unsigned int n_threads, const bool local_approx,
  const arma::vec initial_mode, const unsigned int max_iter, const double conv_tol,
  const unsigned int simulation_method, const bool const_sim, const int model_type,
  const arma::uvec& Z_ind, const arma::uvec& T_ind, const arma::uvec& R_ind) {

  arma::vec a1 = Rcpp::as<arma::vec>(model_["a1"]);
  unsigned int m = a1.n_elem;
  unsigned int n;

  if(model_type > 0) {
    arma::vec y = Rcpp::as<arma::vec>(model_["y"]);
    n = y.n_elem;
  } else {
    arma::vec y = Rcpp::as<arma::mat>(model_["y"]);
    n = y.n_rows;
  }

  ung_amcmc mcmc_run(prior_types, prior_pars, n_iter, n_burnin, n_thin, n, m,
    target_acceptance, gamma, S, true);

  switch (model_type) {
  case 1: {
    ung_ssm model(clone(model_), seed, Z_ind, T_ind, R_ind);
    mcmc_run.approx_mcmc(model, end_ram, local_approx, initial_mode,
      max_iter, conv_tol);
    if(nsim_states > 1) {
    switch (simulation_method) {
    case 1:
      mcmc_run.is_correction_psi(model, nsim_states, const_sim, n_threads);
      break;
    case 2:
      mcmc_run.is_correction_bsf(model, nsim_states, const_sim, n_threads);
      break;
    case 3:
      mcmc_run.is_correction_spdk(model, nsim_states, const_sim, n_threads);
      break;
    }
    } else {
      if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
    }
  } break;
  case 2: {
    ung_bsm model(clone(model_), seed);
    mcmc_run.approx_mcmc(model, end_ram, local_approx, initial_mode,
      max_iter, conv_tol);
    if(nsim_states > 1) {
    switch (simulation_method) {
    case 1:
      mcmc_run.is_correction_psi(model, nsim_states, const_sim, n_threads);
      break;
    case 2:
      mcmc_run.is_correction_bsf(model, nsim_states, const_sim, n_threads);
      break;
    case 3:
      mcmc_run.is_correction_spdk(model, nsim_states, const_sim, n_threads);
      break;
    }
    } else {
      if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
    }
  } break;
  case 3: {
    ung_svm model(clone(model_), seed);
    mcmc_run.approx_mcmc(model, end_ram, local_approx, initial_mode,
      max_iter, conv_tol);
    if(nsim_states > 1) {
    switch (simulation_method) {
    case 1:
      mcmc_run.is_correction_psi(model, nsim_states, const_sim, n_threads);
      break;
    case 2:
      mcmc_run.is_correction_bsf(model, nsim_states, const_sim, n_threads);
      break;
    case 3:
      mcmc_run.is_correction_spdk(model, nsim_states, const_sim, n_threads);
      break;
    }
    } else {
      if(nsim_states == 1) mcmc_run.approx_state_posterior(model, n_threads);
    }
  } break;
  }

  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("weights") = mcmc_run.weight_storage,
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,
    Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}

// [[Rcpp::export]]
Rcpp::List nonlinear_pm_mcmc(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_,
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int max_iter, const double conv_tol,
  const unsigned int simulation_method, const unsigned int iekf_iter) {

  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_,
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);

  mcmc mcmc_run(arma::uvec(theta.n_elem), arma::mat(1,1), n_iter, n_burnin, n_thin, model.n,
    model.m, target_acceptance, gamma, S, true);

  switch (simulation_method) {
  case 1:
    mcmc_run.pm_mcmc_psi_nlg(model, end_ram, nsim_states, max_iter, conv_tol, iekf_iter);
    break;
  case 2:
    mcmc_run.pm_mcmc_bsf_nlg(model, end_ram, nsim_states);
    break;
  }


  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,
    Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}
// [[Rcpp::export]]
Rcpp::List nonlinear_da_mcmc(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_,
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads,
  const unsigned int max_iter, const double conv_tol,
  const unsigned int simulation_method, const unsigned int iekf_iter) {

  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_,
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);

  mcmc mcmc_run(arma::uvec(theta.n_elem), arma::mat(1,1), n_iter, n_burnin, n_thin, model.n,
    model.m, target_acceptance, gamma, S, true);


  switch (simulation_method) {
  case 1:
    mcmc_run.da_mcmc_psi_nlg(model, end_ram, nsim_states, max_iter, conv_tol, iekf_iter);
    break;
  case 2:
    mcmc_run.da_mcmc_bsf_nlg(model, end_ram, nsim_states, max_iter, conv_tol, iekf_iter);
    break;
  }


  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}

// [[Rcpp::export]]
Rcpp::List nonlinear_ekf_mcmc(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_,
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int max_iter, const double conv_tol
  , const unsigned int n_threads, const unsigned int iekf_iter) {

  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_,
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);

  nlg_amcmc mcmc_run(arma::uvec(theta.n_elem), arma::mat(1,1), n_iter, n_burnin, n_thin, model.n,
    model.m, target_acceptance, gamma, S, 1);

  mcmc_run.approx_mcmc(model, max_iter, conv_tol, end_ram, iekf_iter);

  mcmc_run.gaussian_sampling(model, n_threads);
  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}

// [[Rcpp::export]]
Rcpp::List nonlinear_is_mcmc(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_,
  SEXP T_fn_, SEXP R_fn_, SEXP Z_gn_, SEXP T_gn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta, SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params, const arma::uvec& time_varying,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int nsim_states, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads, const bool const_sim,
  const unsigned int simulation_method, const unsigned int max_iter,
  const double conv_tol, const unsigned int iekf_iter) {

  nlg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, Z_gn_, T_gn_, a1_fn_, P1_fn_,
    theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    time_varying, seed);

  nlg_amcmc mcmc_run(arma::uvec(theta.n_elem), arma::mat(1,1), n_iter, n_burnin, n_thin, model.n,
    model.m, target_acceptance, gamma, S, simulation_method == 1);

  mcmc_run.approx_mcmc(model, max_iter, conv_tol, end_ram, iekf_iter);
  if (simulation_method == 1) {
    mcmc_run.is_correction_psi(model, nsim_states, const_sim, n_threads);
  } else {
    mcmc_run.is_correction_bsf(model, nsim_states, const_sim, n_threads);
  }
  return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
    Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
    Rcpp::Named("weights") = mcmc_run.weight_storage,
    Rcpp::Named("counts") = mcmc_run.count_storage,
    Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
    Rcpp::Named("S") = mcmc_run.S,
    Rcpp::Named("posterior") = mcmc_run.posterior_storage);
}

// [[Rcpp::export]]
Rcpp::List general_gaussian_mcmc(const arma::mat& y, SEXP Z_fn_, SEXP H_fn_,
  SEXP T_fn_, SEXP R_fn_, SEXP a1_fn_, SEXP P1_fn_,
  const arma::vec& theta,
  SEXP D_fn_, SEXP C_fn_,
  SEXP log_prior_pdf_, const arma::vec& known_params,
  const arma::mat& known_tv_params,
  const unsigned int n_states, const unsigned int n_etas,
  const unsigned int seed, const unsigned int n_iter,
  const unsigned int n_burnin, const unsigned int n_thin,
  const double gamma, const double target_acceptance, const arma::mat S,
  const bool end_ram, const unsigned int n_threads, const bool sim_states) {

  lgg_ssm model(y, Z_fn_, H_fn_, T_fn_, R_fn_, a1_fn_, P1_fn_,
    D_fn_, C_fn_, theta, log_prior_pdf_, known_params, known_tv_params, n_states, n_etas,
    seed);

  mcmc mcmc_run(arma::uvec(theta.n_elem), arma::mat(1,1), n_iter, n_burnin, n_thin,
    model.n, model.m, target_acceptance, gamma, S, true);

  mcmc_run.mcmc_gaussian(model, end_ram);
  if(sim_states) mcmc_run.state_posterior(model, n_threads);

  if(sim_states) {
    return Rcpp::List::create(Rcpp::Named("alpha") = mcmc_run.alpha_storage,
      Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  } else {
    return Rcpp::List::create(Rcpp::Named("theta") = mcmc_run.theta_storage.t(),
      Rcpp::Named("counts") = mcmc_run.count_storage,
      Rcpp::Named("acceptance_rate") = mcmc_run.acceptance_rate,
      Rcpp::Named("S") = mcmc_run.S,  Rcpp::Named("posterior") = mcmc_run.posterior_storage);
  }
}
